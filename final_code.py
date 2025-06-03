# -*-coding:utf-8-*-
import cv2
import time
import smbus
import numpy as np
from robomaster import robot
from robomaster import camera
from ultralytics import YOLO

# I2C地址
I2C_ADDR = 0x34  # I2C address
ASR_RESULT_ADDR = 100  # ASR speech recognition result register address (0x64)
ASR_SPEAK_ADDR = 110  # ASR broadcasting setting register address (0x6E)
ASR_CMDMAND = 0x00  # Type of broadcast language: Command words
ASR_ANNOUNCER = 0xFF  # Type of announcer: Ordinary announcer


class ASRModule:
    def __init__(self, address, bus=1):
        # Initialize the I2C bus and device addresses
        self.bus = smbus.SMBus(bus)  # Use I2C bus 1
        self.address = address  # The I2C address of the device
        self.send = [0, 0]  # Initialize the list of sent data

    def wire_write_byte(self, val):
        """
        Write a single byte to the device
        :param val: The byte value to be written
        :return: If the write is successful, it returns True; if it fails, it returns False
        """
        try:
            self.bus.write_byte(self.address, val)  # Send bytes to the device
            return True  # Write successfully
        except IOError:
            return False  # Write failed and return False

    def wire_write_data_array(self, reg, val, length):
        """
        Write the byte list to the specified register
        :param reg: Register address
        :param val: The list of bytes to be written
        :param length: The number of bytes to be written
        :return: If the write is successful, it returns True; if it fails, it returns False
        """
        try:
            self.bus.write_i2c_block_data(self.address, reg, val[:length])  # Send the byte list to the specified register of the device
            return True  # Write successfully
        except IOError:
            return False  # Write failed and return False

    def wire_read_data_array(self, reg, length):
        """
        Read the byte list from the specified register
        :param reg: Register address
        :param length: The number of bytes to be read
        :return: The list of bytes read, and an empty list is returned when it fails
        """
        try:
            result = self.bus.read_i2c_block_data(self.address, reg, length)  # Read the byte list from the device
            return result  # Return the reading result
        except IOError:
            return []  # Reading failed and an empty list was returned

    def rec_recognition(self):
        """
        识别结果读取
        :return: Identify the result. If the reading fails, return 0
        """
        result = self.wire_read_data_array(ASR_RESULT_ADDR, 1)  # Read a byte from the result register
        if result:
            return result  # Return the read result
        return 0  # If there is no result, return 0

    def speak(self, cmd, id):
        """
        Send speaking commands to the device
        :param cmd: command byte
        :param id: The speaking ID
        """
        if cmd == ASR_ANNOUNCER or cmd == ASR_CMDMAND:  # Check whether the command is valid
            self.send[0] = cmd  # Set the first element of the sending list as the command
            self.send[1] = id  # Set the second element of the sending list as ID
            self.wire_write_data_array(ASR_SPEAK_ADDR, self.send, 2)  # Send the command and ID to the specified register


class SportsBallTracker:
    def __init__(self):
        self.a_flag = False  # Initialize here
        self.running = False
        self.max_rotate_angle = 30    # Maximum single adjustment Angle (degree)
        self.last_offset = 0          # Record the direction of the last offset
        self.movement_history = []    # Used for analyzing motion trajectories
        # Initialize YOLOv8 model
        try:
            self.model = YOLO('best12.pt')  # Load the custom model
        except Exception as e:
            print(f"Model loading failed.: {e}")
            exit(1)

        self.target_class_id = 80  # ID
        self.center_threshold = 0.1  # Central area threshold (10% of the picture width)

        # Initialize the robot
        self.ep_robot = robot.Robot()
        self.ep_robot.initialize(conn_type="rndis")  # Adjust according to the actual connection method

        self.ep_chassis = self.ep_robot.chassis
        self.ep_camera = self.ep_robot.camera
        self.ep_gripper = self.ep_robot.gripper
        self.ep_arm = self.ep_robot.robotic_arm

        # motion parameter
        self.rotate_speed = 30  # Rotation speed (degrees per second)
        self.forward_speed = 0.3  # Forward speed (m/s)
        self.max_rotate_angle = 45  # Maximum single rotation Angle

    def adjust_position(self, frame, box):
        """
        Adjust the chassis direction according to the position of the target object (Improved version)
        """
        frame_width = frame.shape[1]
        box_center_x = (box[0] + box[2]) / 2
        offset_ratio = (box_center_x - frame_width / 2) / (frame_width / 2)  # [-1, 1]

        # Determine the continuous direction offset to avoid misoperation
        if not hasattr(self, 'confirmed_count'):
            self.confirmed_count = 0

        if np.sign(offset_ratio) == np.sign(getattr(self, 'last_offset_ratio', 0)):
            self.confirmed_count += 1
        else:
            self.confirmed_count = 1

        self.last_offset_ratio = offset_ratio

        #if self.confirmed_count < 3:
         #   print(f"方向不稳定，等待确认中（{self.confirmed_count}/3）")
          #  return 0.0

        # Tolerance in the central area, avoiding small jitter and repeated adjustments
        dead_zone = 0.03
        if abs(offset_ratio) < dead_zone:
            print(f"The object is centered and shifted: {offset_ratio * 100:.1f}%")
            self.a_flag = True
            return 0.0

        # Nonlinear control: Faster response to large offsets
        control_ratio = np.sign(offset_ratio) * (abs(offset_ratio) ** 0.7)

        # Control angular velocity
        min_speed = 10
        max_speed = 30
        rotate_speed = min_speed + (max_speed - min_speed) * abs(control_ratio)

        # Rotation Angle, limited range
        target_angle = np.clip(-control_ratio * self.max_rotate_angle,
                               -self.max_rotate_angle, self.max_rotate_angle)

        print(f"Direction shifted: {offset_ratio * 100:.1f}% | Adjust angle: {target_angle:.1f}° | Speed: {rotate_speed:.1f}°/s")
        
        final_angle = 0.0
        # Rotate in segments to reduce shake
        if abs(target_angle) > 20:
            intermediate_angle = target_angle * 0.5
            self.ep_chassis.move(x=0, y=0, z=intermediate_angle, z_speed=rotate_speed).wait_for_completed()
            final_angle += intermediate_angle
            time.sleep(0.2)

        self.ep_chassis.move(x=0, y=0, z=target_angle, z_speed=rotate_speed).wait_for_completed()
        final_angle += target_angle
        return final_angle

    def estimate_distance(self, box, real_width_cm=7.0, focal_length_px=400):
        """
        Estimate the distance based on the detection box (unit: cm)
        Parameter:
            box: [x1, y1, x2, y2]
            real_width_cm: The actual width of the object (such as the diameter of a ball, in cm)
            focal_length_px: Camera focal length (pixels, requires calibration)
        Return:
            Distance (cm)
        """
        pixel_width = box[2] - box[0]
        if pixel_width == 0:
            return float('inf')  # Prevent division by 0
        distance_cm = (real_width_cm * focal_length_px) / pixel_width
        return distance_cm

    def run(self):
        asr_module = ASRModule(I2C_ADDR)
        while True:
            recognition_result = asr_module.rec_recognition()
            if recognition_result[0] != 0:
                if recognition_result[0] == 5:
                    print("Begin.")
                    self.running = True  # Set the running flag to True
                    break

        cnt_number = 0
        # Take the box at the very beginning
        # Open the mechanical claws
        self.ep_gripper.open(power=150)
        time.sleep(1)
        self.ep_gripper.open(power=150)
        time.sleep(1)
        self.ep_gripper.pause()
        time.sleep(1)
        time.sleep(1)

        # Take the box
        # Closed mechanical claw
        self.ep_gripper.close(power=10)
        time.sleep(1)
        self.ep_gripper.pause()

        try:
            # Use the built-in camera of the robot
            self.ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)
            detection_count = 0
            max_detection_count = 3
  
            while self.running:
                asr_module_1 = ASRModule(I2C_ADDR)
                recognition_result_1 = asr_module_1.rec_recognition()
                if recognition_result_1[0] != 0:
                    if recognition_result_1[0] == 6:
                        print("Stop.")
                        self.running = False
                        break

                frame = self.ep_camera.read_cv2_image(strategy="newest")
                if frame is None:
                    continue

                # YOLO Detection
                results = self.model(frame, verbose=False)

                # Process the test results
                target_detected = False
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        class_id = int(box.cls)
                        conf = float(box.conf)

                        # The target object was detected and the confidence level was >10%
                        if class_id == self.target_class_id and conf > 0.1:
                            target_detected = True
                            # Obtain the coordinates of the detection box
                            xyxy = box.xyxy[0].cpu().numpy()
                            # Adjust position
                            # self.adjust_position(frame, xyxy)
                            break
                if target_detected:
                    detection_count += 1
                else:
                    detection_count = 0
                if detection_count >= max_detection_count:
                    print("Object detected. Stop moving.")
                    time.sleep(1)
                    last_angle = 0.0
                    angle = 0.0
                    while self.running:
                        frame = self.ep_camera.read_cv2_image(strategy="newest")
                        results = self.model(frame, verbose=False)

                        for result in results:
                            for box in result.boxes:
                                if int(box.cls) == self.target_class_id and float(box.conf) > 0.4:
                                    xyxy = box.xyxy[0].cpu().numpy()
                                    angle += self.adjust_position(frame, xyxy)
                                    
                        if self.a_flag:
                            self.a_flag = False
                            break
                        time.sleep(1)
                        frame = self.ep_camera.read_cv2_image(strategy="newest")
                        time.sleep(1)
                        results = self.model(frame, verbose=False)
                        for result in results:
                            boxes = result.boxes
                            for box in boxes:
                                if int(box.cls) == self.target_class_id and float(box.conf) > 0.1:
                                    xyxy = box.xyxy[0].cpu().numpy()
                                    break
                    last = 0.0
                    cnt = 0
                    cnt_dis_num = 0
                    # Adjust distance
                    time.sleep(1)
                    distance = self.estimate_distance(xyxy)
                    print(f"Disance: {distance:.2f}cm，Keep closer...")
                    fl = 1
                    # Adjust distance
                    while self.running and distance > 45:
                        print(f"Last distance: {last}cm，Keep closer...")
                        print(cnt)
                        print(f"diff = {distance - last}")
                        if fl == 0:
                            cnt += 1
                            print(cnt)
                        if cnt > 0:
                            break
                        fl = 0
                        print(f"Distance: {distance}cm，Keep closer...")
                        cnt_dis_num += 1
                        self.ep_chassis.move(x=0.07, y=0, z=0, xy_speed=0.1).wait_for_completed()
                        time.sleep(1)
                        frame = self.ep_camera.read_cv2_image(strategy="newest")
                        time.sleep(1)
                        results = self.model(frame, verbose=False)
                        for result in results:
                            boxes = result.boxes
                            for box in boxes:
                                if int(box.cls) == self.target_class_id and float(box.conf) > 0.1:
                                    xyxy = box.xyxy[0].cpu().numpy()
                                    last = distance
                                    time.sleep(1)
                                    distance = self.estimate_distance(xyxy)
                                    time.sleep(1)
                                    fl = 1
                                    break
                    print(f"Distance: {distance:.2f}cm，Keep closer...")
                    print("The position adjustment is completed. It's time to move forward.")
                    # self.ep_chassis.move(x=0.5, y=0, z=0, xy_speed=0.1).wait_for_completed()
                    # self.ep_chassis.move(x=0.4, y=0, z=0, xy_speed=0.1).wait_for_completed()
                    self.ep_chassis.move(x=0.1, y=0, z=0, xy_speed=0.1).wait_for_completed()
                    print("The position adjustment is completed. Start grasping...")

                    # === Turn 45° to the right to reach the grasping posture position ===
                    self.ep_chassis.move(x=0, y=0, z=-45, z_speed=45).wait_for_completed()
                    time.sleep(0.5)

                    # === The mechanical arm extends and presses down to the target position ===
                    self.ep_arm.move(x=100, y=-250).wait_for_completed()  # Merge the two lateral movements and downward presses
                    time.sleep(0.5)

                    # === Open the gripper and get ready to grasp ===
                    self.ep_gripper.open(power=40)
                    time.sleep(1)
                    self.ep_gripper.pause()

                    # === Lift the object ===
                    self.ep_arm.move(x=-100, y=250).wait_for_completed()
                    time.sleep(0.5)

                    # === Straighten the chassis and turn left by 45 degrees ===
                    self.ep_chassis.move(x=0, y=0, z=45, z_speed=45).wait_for_completed()
                    time.sleep(0.5)

                    # === Stretch out again to get ready to grasp the new object ===
                    self.ep_arm.move(x=100, y=-250).wait_for_completed()
                    time.sleep(0.5)
                    
                    ###########
                    self.ep_chassis.move(x=0.15, y=0, z=0, xy_speed=0.1).wait_for_completed()
                    time.sleep(0.5)
                    self.ep_chassis.move(x=-0.04, y=0, z=0, xy_speed=0.1).wait_for_completed()
                    ###########

                    # === Close the mechanical claw for grasping ===
                    self.ep_gripper.close(power=90)
                    time.sleep(1)
                    self.ep_gripper.pause()

                    # === Lift the object ===
                    self.ep_arm.move(x=-100, y=250).wait_for_completed()
                    time.sleep(0.5)
                    
                    ###################
                    self.ep_chassis.move(x=-0.12, y=0, z=0, xy_speed=0.1).wait_for_completed()
                    #####################
                    
                    # === Turn 45 degrees to the right and put the object into the box ===
                    self.ep_chassis.move(x=0, y=0, z=-48, z_speed=45).wait_for_completed()
                    time.sleep(0.5)

                    # === Put the object into the box ===
                    self.ep_arm.move(x=100, y=-30).wait_for_completed()
                    time.sleep(0.3)

                    self.ep_gripper.open(power=150)
                    time.sleep(1)
                    self.ep_arm.move(x=100, y=-50).wait_for_completed()
                    self.ep_gripper.pause()

                    self.ep_arm.move(x=100, y=-200).wait_for_completed()
                    self.ep_gripper.close(power=90)
                    time.sleep(1)
                    self.ep_gripper.pause()

                    # === Finally, retract the robotic arm ===
                    self.ep_arm.move(x=-100, y=250).wait_for_completed()
                    time.sleep(0.5)

                    # === Turn left and go straight ===
                    self.ep_chassis.move(x=0, y=0, z=45, z_speed=45).wait_for_completed()

                    ### Stop ###
                    self.ep_chassis.stop()

                    # Return to original position
                    # Rotate
                    print(f"Adjust the direction. Rotate Angle: {angle:.1f} degree")

                    for i in range(0, cnt_dis_num):
                        self.ep_chassis.move(x=-0.07, y=0, z=0, xy_speed=0.1).wait_for_completed()
                    self.ep_chassis.move(x=0, y=0, z=-angle, z_speed=self.rotate_speed).wait_for_completed()

                    print(angle)

                else:
                    cnt_number += 1
                    print("The target object was not detected. Keep moving forward")
                    if cnt_number == 22:
                        self.ep_chassis.move(x=0, y=0, z=90, z_speed=45).wait_for_completed()
                        time.sleep(0.5)
                    elif cnt_number == 33:
                        time.sleep(0.5)
                        self.ep_chassis.move(x=0, y=0, z=90, z_speed=45).wait_for_completed()
                        time.sleep(0.5)
                    elif cnt_number == 55:
                        time.sleep(0.5)
                        self.ep_chassis.move(x=0, y=0, z=90, z_speed=45).wait_for_completed()
                        time.sleep(0.5)
                        
                    self.ep_chassis.move(x=0.1, y=0, z=0, xy_speed=0.1).wait_for_completed()

                # Display the test results (optional)
                # annotated_frame = results[0].plot()
                # cv2.imshow("YOLOv8 Tracking", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.ep_camera.stop_video_stream()
            cv2.destroyAllWindows()
            self.ep_chassis.stop()
            self.ep_robot.close()

        def stop(self):
            """Stop moving"""
            self.running = False


if __name__ == '__main__':
    tracker = SportsBallTracker()
    tracker.run()
