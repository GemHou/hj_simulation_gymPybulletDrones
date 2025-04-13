import os
from sys import platform
import time
import math
import pybullet
import collections
from datetime import datetime
import xml.etree.ElementTree as etxml
import pkg_resources
from PIL import Image
# import pkgutil
# egl = pkgutil.get_loader('eglRenderer')
import numpy as np
import pybullet as pbl
import pybullet_data
import gymnasium as gym
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ImageType

SCENARIO = "Arch"  # "None" "Farm" "Arch"
GLOBAL_SCALING = 1


class BaseAviary(gym.Env):
    """Base class for "drone aviary" Gym environments."""

    # metadata = {'render.modes': ['human']}

    ################################################################################

    def hj_random_init(self, percent):
        # stage 1
        x = np.random.uniform(-10, 10) * percent ** 0.5
        y = np.random.uniform(-10, 10) * percent ** 0.5
        z = 1 + (np.random.uniform(-0.5, 0) + np.random.uniform(0, 10)) * percent ** 0.5
        # stage 2
        # x = np.random.uniform(-4, 4)
        # y = np.random.uniform(-4, 4)
        # z = np.random.uniform(2, 4)
        # x = np.random.uniform(-7, 7)
        # y = np.random.uniform(-7, 7)
        # z = np.random.uniform(2, 5)
        # stage 3
        # x = np.random.uniform(-50, 50) * percent ** 0.5
        # y = np.random.uniform(-50, 50) * percent ** 0.5
        # z = 3 + np.random.uniform(0, 20) * percent ** 0.5
        # x = np.random.uniform(-15, 15)
        # y = np.random.uniform(-15, 15)
        # z = np.random.uniform(2, 10)
        # x = -50 + np.random.uniform(-1, 1)
        # y = -50 + np.random.uniform(-2, 2)
        # z = 20 + np.random.uniform(4, 5)
        initial_xyzs = np.vstack([[x, y, z]])
        return initial_xyzs

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 1,
                 neighbourhood_radius: float = np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 vision_attributes=False,
                 output_folder='results'
                 ):
        """Initialization of a generic aviary environment.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obstacles : bool, optional
            Whether to add obstacles to the simulation.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.
        vision_attributes : bool, optional
            Whether to allocate the attributes needed by vision-based aviary subclasses.

        """
        #### Constants #############################################
        self.G = 9.8
        self.RAD2DEG = 180 / np.pi
        self.DEG2RAD = np.pi / 180
        self.CTRL_FREQ = ctrl_freq
        self.PYB_FREQ = pyb_freq
        if self.PYB_FREQ % self.CTRL_FREQ != 0:
            raise ValueError('[ERROR] in BaseAviary.__init__(), pyb_freq is not divisible by env_freq.')
        self.PYB_STEPS_PER_CTRL = int(self.PYB_FREQ / self.CTRL_FREQ)
        self.CTRL_TIMESTEP = 1. / self.CTRL_FREQ
        self.PYB_TIMESTEP = 1. / self.PYB_FREQ
        #### Parameters ############################################
        self.NUM_DRONES = num_drones
        self.NEIGHBOURHOOD_RADIUS = neighbourhood_radius
        #### Options ###############################################
        self.DRONE_MODEL = drone_model
        self.GUI = gui
        self.RECORD = record
        self.PHYSICS = physics
        self.OBSTACLES = obstacles
        self.USER_DEBUG = user_debug_gui
        self.URDF = self.DRONE_MODEL.value + ".urdf"
        self.OUTPUT_FOLDER = output_folder
        #### Load the drone properties from the .urdf file #########
        self.M, \
            self.L, \
            self.THRUST2WEIGHT_RATIO, \
            self.J, \
            self.J_INV, \
            self.KF, \
            self.KM, \
            self.COLLISION_H, \
            self.COLLISION_R, \
            self.COLLISION_Z_OFFSET, \
            self.MAX_SPEED_KMH, \
            self.GND_EFF_COEFF, \
            self.PROP_RADIUS, \
            self.DRAG_COEFF, \
            self.DW_COEFF_1, \
            self.DW_COEFF_2, \
            self.DW_COEFF_3 = self._parseURDFParameters()
        print(
            "[INFO] BaseAviary.__init__() loaded parameters from the drone's .urdf:\n[INFO] m {:f}, L {:f},\n[INFO] ixx {:f}, iyy {:f}, izz {:f},\n[INFO] kf {:f}, km {:f},\n[INFO] t2w {:f}, max_speed_kmh {:f},\n[INFO] gnd_eff_coeff {:f}, prop_radius {:f},\n[INFO] drag_xy_coeff {:f}, drag_z_coeff {:f},\n[INFO] dw_coeff_1 {:f}, dw_coeff_2 {:f}, dw_coeff_3 {:f}".format(
                self.M, self.L, self.J[0, 0], self.J[1, 1], self.J[2, 2], self.KF, self.KM, self.THRUST2WEIGHT_RATIO,
                self.MAX_SPEED_KMH, self.GND_EFF_COEFF, self.PROP_RADIUS, self.DRAG_COEFF[0], self.DRAG_COEFF[2],
                self.DW_COEFF_1, self.DW_COEFF_2, self.DW_COEFF_3))
        #### Compute constants #####################################
        self.GRAVITY = self.G * self.M
        self.HOVER_RPM = np.sqrt(self.GRAVITY / (4 * self.KF))
        self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO * self.GRAVITY) / (4 * self.KF))
        self.MAX_THRUST = (4 * self.KF * self.MAX_RPM ** 2)
        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MAX_XY_TORQUE = (2 * self.L * self.KF * self.MAX_RPM ** 2) / np.sqrt(2)
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MAX_XY_TORQUE = (self.L * self.KF * self.MAX_RPM ** 2)
        elif self.DRONE_MODEL == DroneModel.RACE:
            self.MAX_XY_TORQUE = (2 * self.L * self.KF * self.MAX_RPM ** 2) / np.sqrt(2)
        self.MAX_Z_TORQUE = (2 * self.KM * self.MAX_RPM ** 2)
        self.GND_EFF_H_CLIP = 0.25 * self.PROP_RADIUS * np.sqrt(
            (15 * self.MAX_RPM ** 2 * self.KF * self.GND_EFF_COEFF) / self.MAX_THRUST)
        #### Create attributes for vision tasks ####################
        if self.RECORD:
            self.ONBOARD_IMG_PATH = os.path.join(self.OUTPUT_FOLDER,
                                                 "recording_" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
            os.makedirs(os.path.dirname(self.ONBOARD_IMG_PATH), exist_ok=True)
        self.VISION_ATTR = vision_attributes
        if self.VISION_ATTR:
            self.IMG_RES = np.array([64, 48])
            self.IMG_FRAME_PER_SEC = 24
            self.IMG_CAPTURE_FREQ = int(self.PYB_FREQ / self.IMG_FRAME_PER_SEC)
            self.rgb = np.zeros(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4)))
            self.dep = np.ones(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0])))
            self.seg = np.zeros(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0])))
            if self.IMG_CAPTURE_FREQ % self.PYB_STEPS_PER_CTRL != 0:
                print(
                    "[ERROR] in BaseAviary.__init__(), PyBullet and control frequencies incompatible with the desired video capture frame rate ({:f}Hz)".format(
                        self.IMG_FRAME_PER_SEC))
                exit()
            if self.RECORD:
                for i in range(self.NUM_DRONES):
                    os.makedirs(os.path.dirname(self.ONBOARD_IMG_PATH + "/drone_" + str(i) + "/"), exist_ok=True)
        #### Connect to PyBullet ###################################
        if self.GUI:
            #### With debug GUI ########################################
            self.CLIENT = pbl.connect(pbl.GUI)  # p.connect(p.GUI, options="--opengl2")
            for i in [pbl.COV_ENABLE_RGB_BUFFER_PREVIEW, pbl.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                      pbl.COV_ENABLE_SEGMENTATION_MARK_PREVIEW]:
                pbl.configureDebugVisualizer(i, 0, physicsClientId=self.CLIENT)
            pbl.resetDebugVisualizerCamera(cameraDistance=3,
                                           cameraYaw=-30,
                                           cameraPitch=-30,
                                           cameraTargetPosition=[0, 0, 0],
                                           physicsClientId=self.CLIENT
                                           )
            ret = pbl.getDebugVisualizerCamera(physicsClientId=self.CLIENT)
            print("viewMatrix", ret[2])
            print("projectionMatrix", ret[3])
            if self.USER_DEBUG:
                #### Add input sliders to the GUI ##########################
                self.SLIDERS = -1 * np.ones(4)
                for i in range(4):
                    self.SLIDERS[i] = pbl.addUserDebugParameter("Propeller " + str(i) + " RPM", 0, self.MAX_RPM,
                                                                self.HOVER_RPM, physicsClientId=self.CLIENT)
                self.INPUT_SWITCH = pbl.addUserDebugParameter("Use GUI RPM", 9999, -1, 0, physicsClientId=self.CLIENT)
        else:
            #### Without debug GUI #####################################
            self.CLIENT = pbl.connect(pbl.DIRECT)
            #### Uncomment the following line to use EGL Render Plugin #
            #### Instead of TinyRender (CPU-based) in PYB's Direct mode
            # if platform == "linux": p.setAdditionalSearchPath(pybullet_data.getDataPath()); plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin"); print("plugin=", plugin)
            if self.RECORD:
                #### Set the camera parameters to save frames in DIRECT mode
                self.VID_WIDTH = int(640)
                self.VID_HEIGHT = int(480)
                self.FRAME_PER_SEC = 24
                self.CAPTURE_FREQ = int(self.PYB_FREQ / self.FRAME_PER_SEC)
                self.CAM_VIEW = pbl.computeViewMatrixFromYawPitchRoll(distance=3,
                                                                      yaw=-30,
                                                                      pitch=-30,
                                                                      roll=0,
                                                                      cameraTargetPosition=[0, 0, 0],
                                                                      upAxisIndex=2,
                                                                      physicsClientId=self.CLIENT
                                                                      )
                self.CAM_PRO = pbl.computeProjectionMatrixFOV(fov=60.0,
                                                              aspect=self.VID_WIDTH / self.VID_HEIGHT,
                                                              nearVal=0.1,
                                                              farVal=1000.0
                                                              )
        #### Set initial poses #####################################
        if initial_xyzs is None:
            self.INIT_XYZS = np.vstack([np.array([x * 4 * self.L for x in range(self.NUM_DRONES)]), \
                                        np.array([y * 4 * self.L for y in range(self.NUM_DRONES)]), \
                                        np.ones(self.NUM_DRONES) * (
                                                self.COLLISION_H / 2 - self.COLLISION_Z_OFFSET + .1)]).transpose().reshape(
                self.NUM_DRONES, 3)
        elif np.array(initial_xyzs).shape == (self.NUM_DRONES, 3):
            self.INIT_XYZS = initial_xyzs
        else:
            print("np.array(initial_xyzs).shape: ", np.array(initial_xyzs).shape)
            print("[ERROR] invalid initial_xyzs in BaseAviary.__init__(), try initial_xyzs.reshape(NUM_DRONES,3)")
        if initial_rpys is None:
            self.INIT_RPYS = np.zeros((self.NUM_DRONES, 3))
        elif np.array(initial_rpys).shape == (self.NUM_DRONES, 3):
            self.INIT_RPYS = initial_rpys
        else:
            print("[ERROR] invalid initial_rpys in BaseAviary.__init__(), try initial_rpys.reshape(NUM_DRONES,3)")
        #### Create action and observation spaces ##################
        self.action_space = self._actionSpace()
        self.observation_space = self._observationSpace()
        #### Housekeeping ##########################################
        self._housekeeping(percent=0)
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Start video recording #################################
        self._startVideoRecording()

    ################################################################################

    def reset(self, percent,
              seed: int = None,
              options: dict = None):
        """Resets the environment.

        Parameters
        ----------
        seed : int, optional
            Random seed.
        options : dict[..], optional
            Additinonal options, unused

        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """

        # TODO : initialize random number generator with seed

        pbl.resetSimulation(physicsClientId=self.CLIENT)
        #### Housekeeping ##########################################
        self._housekeeping(percent)
        # 创建一个视觉形状（球体）
        sphere_visual_shape_start = pybullet.createVisualShape(shapeType=pybullet.GEOM_SPHERE,
                                                  radius=0.1,
                                                  rgbaColor=[0, 0, 1, 0.5])

        # 创建一个刚体，仅使用视觉形状，不创建碰撞形状，质量设为 0
        self.target_sphere_start = pybullet.createMultiBody(baseMass=0,
                                        baseInertialFramePosition=[0, 0, 0],
                                        baseVisualShapeIndex=sphere_visual_shape_start,
                                        basePosition=[self.INIT_XYZS[0, 0], self.INIT_XYZS[0, 1], self.INIT_XYZS[0, 2]],
                                        baseCollisionShapeIndex=-1)
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Start video recording #################################
        self._startVideoRecording()
        #### Return the initial observation ########################
        self.target_x = np.random.uniform(-10, 10) * percent
        self.target_y = np.random.uniform(-10, 10) * percent
        self.target_z = 2.5 + np.random.uniform(-1.5, 1.5) * percent
        self.target_x_vel = np.random.uniform(-0.03, 0.03) * percent
        self.target_y_vel = np.random.uniform(-0.03, 0.03) * percent
        self.target_z_vel = np.random.uniform(-0.01, 0.01) * percent
        initial_obs = self._computeObs()
        initial_info = self._computeInfo()

        # 创建一个视觉形状（球体）
        sphere_visual_shape = pybullet.createVisualShape(shapeType=pybullet.GEOM_SPHERE,
                                                  radius=0.1,
                                                  rgbaColor=[1, 0, 0, 0.5])

        # 创建一个刚体，仅使用视觉形状，不创建碰撞形状，质量设为 0
        self.target_sphere = pybullet.createMultiBody(baseMass=0,
                                        baseInertialFramePosition=[0, 0, 0],
                                        baseVisualShapeIndex=sphere_visual_shape,
                                        basePosition=[self.target_x, self.target_y, self.target_z],
                                        baseCollisionShapeIndex=-1)

        return initial_obs, initial_info

    ################################################################################

    def step(self,
             action
             ):
        """Advances the environment by one simulation step.

        Parameters
        ----------
        action : ndarray | dict[..]
            The input action for one or more drones, translated into RPMs by
            the specific implementation of `_preprocessAction()` in each subclass.

        Returns
        -------
        ndarray | dict[..]
            The step's observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        float | dict[..]
            The step's reward value(s), check the specific implementation of `_computeReward()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current episode is over, check the specific implementation of `_computeTerminated()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current episode is truncated, check the specific implementation of `_computeTruncated()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current episode is trunacted, always false.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """
        action = np.clip(action, a_min=-5, a_max=5)
        #### Save PNG video frames if RECORD=True and GUI=False ####
        if self.RECORD and not self.GUI and self.step_counter % self.CAPTURE_FREQ == 0:
            [w, h, rgb, dep, seg] = pbl.getCameraImage(width=self.VID_WIDTH,
                                                       height=self.VID_HEIGHT,
                                                       shadow=1,
                                                       viewMatrix=self.CAM_VIEW,
                                                       projectionMatrix=self.CAM_PRO,
                                                       renderer=pbl.ER_TINY_RENDERER,
                                                       flags=pbl.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                                       physicsClientId=self.CLIENT
                                                       )
            (Image.fromarray(np.reshape(rgb, (h, w, 4)), 'RGBA')).save(
                os.path.join(self.IMG_PATH, "frame_" + str(self.FRAME_NUM) + ".png"))
            #### Save the depth or segmentation view instead #######
            # dep = ((dep-np.min(dep)) * 255 / (np.max(dep)-np.min(dep))).astype('uint8')
            # (Image.fromarray(np.reshape(dep, (h, w)))).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
            # seg = ((seg-np.min(seg)) * 255 / (np.max(seg)-np.min(seg))).astype('uint8')
            # (Image.fromarray(np.reshape(seg, (h, w)))).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
            self.FRAME_NUM += 1
            if self.VISION_ATTR:
                for i in range(self.NUM_DRONES):
                    self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i)
                    #### Printing observation to PNG frames example ############
                    self._exportImage(img_type=ImageType.RGB,  # ImageType.BW, ImageType.DEP, ImageType.SEG
                                      img_input=self.rgb[i],
                                      path=self.ONBOARD_IMG_PATH + "/drone_" + str(i) + "/",
                                      frame_num=int(self.step_counter / self.IMG_CAPTURE_FREQ)
                                      )
        #### Read the GUI's input parameters #######################
        if self.GUI and self.USER_DEBUG:
            current_input_switch = pbl.readUserDebugParameter(self.INPUT_SWITCH, physicsClientId=self.CLIENT)
            if current_input_switch > self.last_input_switch:
                self.last_input_switch = current_input_switch
                self.USE_GUI_RPM = True if self.USE_GUI_RPM == False else False
        if self.USE_GUI_RPM:
            for i in range(4):
                self.gui_input[i] = pbl.readUserDebugParameter(int(self.SLIDERS[i]), physicsClientId=self.CLIENT)
            clipped_action = np.tile(self.gui_input, (self.NUM_DRONES, 1))
            if self.step_counter % (self.PYB_FREQ / 2) == 0:
                self.GUI_INPUT_TEXT = [pbl.addUserDebugText("Using GUI RPM",
                                                            textPosition=[0, 0, 0],
                                                            textColorRGB=[1, 0, 0],
                                                            lifeTime=1,
                                                            textSize=2,
                                                            parentObjectUniqueId=self.DRONE_IDS[i],
                                                            parentLinkIndex=-1,
                                                            replaceItemUniqueId=int(self.GUI_INPUT_TEXT[i]),
                                                            physicsClientId=self.CLIENT
                                                            ) for i in range(self.NUM_DRONES)]
        #### Save, preprocess, and clip the action to the max. RPM #
        else:
            clipped_action = np.reshape(self._preprocessAction(action), (self.NUM_DRONES, 4))
        #### Repeat for as many as the aggregate physics steps #####
        for _ in range(self.PYB_STEPS_PER_CTRL):
            #### Update and store the drones kinematic info for certain
            #### Between aggregate steps for certain types of update ###
            if self.PYB_STEPS_PER_CTRL > 1 and self.PHYSICS in [Physics.DYN, Physics.PYB_GND, Physics.PYB_DRAG,
                                                                Physics.PYB_DW, Physics.PYB_GND_DRAG_DW]:
                self._updateAndStoreKinematicInformation()
            #### Step the simulation using the desired physics update ##
            for i in range(self.NUM_DRONES):
                if self.PHYSICS == Physics.PYB:
                    self._physics(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.DYN:
                    self._dynamics(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_GND:
                    self._physics(clipped_action[i, :], i)
                    self._groundEffect(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_DRAG:
                    self._physics(clipped_action[i, :], i)
                    self._drag(self.last_clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_DW:
                    self._physics(clipped_action[i, :], i)
                    self._downwash(i)
                elif self.PHYSICS == Physics.PYB_GND_DRAG_DW:
                    self._physics(clipped_action[i, :], i)
                    self._groundEffect(clipped_action[i, :], i)
                    self._drag(self.last_clipped_action[i, :], i)
                    self._downwash(i)
            #### PyBullet computes the new state, unless Physics.DYN ###
            if self.PHYSICS != Physics.DYN:
                pbl.stepSimulation(physicsClientId=self.CLIENT)
            #### Save the last applied action (e.g. to compute drag) ###
            self.last_clipped_action = clipped_action
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Prepare the return values #############################
        obs = self._computeObs()
        reward = self._computeReward()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()
        #### Advance the step counter ##############################
        self.step_counter = self.step_counter + (1 * self.PYB_STEPS_PER_CTRL)

        self.target_x += self.target_x_vel
        self.target_y += self.target_y_vel
        self.target_z += self.target_z_vel
        pybullet.resetBasePositionAndOrientation(self.target_sphere, [self.target_x, self.target_y, self.target_z], [0, 0, 0, 1])

        return obs, reward, terminated, truncated, info

    ################################################################################

    def render(self,
               mode='human',
               close=False
               ):
        """Prints a textual output of the environment.

        Parameters
        ----------
        mode : str, optional
            Unused.
        close : bool, optional
            Unused.

        """
        if self.first_render_call and not self.GUI:
            print(
                "[WARNING] BaseAviary.render() is implemented as text-only, re-initialize the environment using Aviary(gui=True) to use PyBullet's graphical interface")
            self.first_render_call = False
        print("\n[INFO] BaseAviary.render() ——— it {:04d}".format(self.step_counter),
              "——— wall-clock time {:.1f}s,".format(time.time() - self.RESET_TIME),
              "simulation time {:.1f}s@{:d}Hz ({:.2f}x)".format(self.step_counter * self.PYB_TIMESTEP, self.PYB_FREQ,
                                                                (self.step_counter * self.PYB_TIMESTEP) / (
                                                                        time.time() - self.RESET_TIME)))
        for i in range(self.NUM_DRONES):
            print("[INFO] BaseAviary.render() ——— drone {:d}".format(i),
                  "——— x {:+06.2f}, y {:+06.2f}, z {:+06.2f}".format(self.pos[i, 0], self.pos[i, 1], self.pos[i, 2]),
                  "——— velocity {:+06.2f}, {:+06.2f}, {:+06.2f}".format(self.vel[i, 0], self.vel[i, 1], self.vel[i, 2]),
                  "——— roll {:+06.2f}, pitch {:+06.2f}, yaw {:+06.2f}".format(self.rpy[i, 0] * self.RAD2DEG,
                                                                              self.rpy[i, 1] * self.RAD2DEG,
                                                                              self.rpy[i, 2] * self.RAD2DEG),
                  "——— angular velocity {:+06.4f}, {:+06.4f}, {:+06.4f} ——— ".format(self.ang_v[i, 0], self.ang_v[i, 1],
                                                                                     self.ang_v[i, 2]))

    ################################################################################

    def close(self):
        """Terminates the environment.
        """
        if self.RECORD and self.GUI:
            pbl.stopStateLogging(self.VIDEO_ID, physicsClientId=self.CLIENT)
        pbl.disconnect(physicsClientId=self.CLIENT)

    ################################################################################

    def getPyBulletClient(self):
        """Returns the PyBullet Client Id.

        Returns
        -------
        int:
            The PyBullet Client Id.

        """
        return self.CLIENT

    ################################################################################

    def getDroneIds(self):
        """Return the Drone Ids.

        Returns
        -------
        ndarray:
            (NUM_DRONES,)-shaped array of ints containing the drones' ids.

        """
        return self.DRONE_IDS

    ################################################################################

    def hj_calc_orientation(self, theta_x, theta_y, theta_z):
        theta_x_rad = math.radians(theta_x)
        theta_y_rad = math.radians(theta_y)
        theta_z_rad = math.radians(theta_z)
        # 计算四元数的各个分量
        w = math.cos(theta_x_rad / 2) * math.cos(theta_y_rad / 2) * math.cos(theta_z_rad / 2) + \
            math.sin(theta_x_rad / 2) * math.sin(theta_y_rad / 2) * math.sin(theta_z_rad / 2)
        x = math.sin(theta_x_rad / 2) * math.cos(theta_y_rad / 2) * math.cos(theta_z_rad / 2) - \
            math.cos(theta_x_rad / 2) * math.sin(theta_y_rad / 2) * math.sin(theta_z_rad / 2)
        y = math.cos(theta_x_rad / 2) * math.sin(theta_y_rad / 2) * math.cos(theta_z_rad / 2) + \
            math.sin(theta_x_rad / 2) * math.cos(theta_y_rad / 2) * math.sin(theta_z_rad / 2)
        z = math.cos(theta_x_rad / 2) * math.cos(theta_y_rad / 2) * math.sin(theta_z_rad / 2) - \
            math.sin(theta_x_rad / 2) * math.sin(theta_y_rad / 2) * math.cos(theta_z_rad / 2)
        return w, x, y, z

    def hj_create_cylinder(self):
        cube_collision_shape = pbl.createCollisionShape(pbl.GEOM_BOX, halfExtents=[10, 10, 0.1])
        cube_visual_shape = pbl.createVisualShape(pbl.GEOM_BOX, halfExtents=[20, 20, 0.1],
                                                  rgbaColor=[96/255, 110/255, 59/255, 1])
        cube_body = pbl.createMultiBody(baseMass=10, baseCollisionShapeIndex=cube_collision_shape,
                                        baseVisualShapeIndex=cube_visual_shape,
                                        basePosition=[0, 0, 0.05])

        cylinder_collision_shape = pbl.createCollisionShape(pbl.GEOM_CYLINDER, radius=0.1, height=6)
        cylinder_visual_shape = pbl.createVisualShape(pbl.GEOM_CYLINDER, radius=0.1, length=6,
                                                      rgbaColor=[123 / 255, 105 / 255, 72 / 255, 1])
        cylinder_body = pbl.createMultiBody(baseMass=1, baseCollisionShapeIndex=cylinder_collision_shape,
                                            baseVisualShapeIndex=cylinder_visual_shape,
                                            basePosition=[8, 0, 3.1],
                                            )  # baseOrientation=[x, y, z, w]
        cylinder_body = pbl.createMultiBody(baseMass=1, baseCollisionShapeIndex=cylinder_collision_shape,
                                            baseVisualShapeIndex=cylinder_visual_shape,
                                            basePosition=[-8, 0, 3.1],
                                            )  # baseOrientation=[x, y, z, w]

        # cylinder_collision_shape = pbl.createCollisionShape(pbl.GEOM_CYLINDER, radius=0.02, height=16)
        # cylinder_visual_shape = pbl.createVisualShape(pbl.GEOM_CYLINDER, radius=0.02, length=16,
        #                                               rgbaColor=[14 / 255, 15 / 255, 17 / 255, 1])
        # w, x, y, z = self.hj_calc_orientation(0, 90, 0)
        # cylinder_body = pbl.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=cylinder_collision_shape,
        #                                     baseVisualShapeIndex=cylinder_visual_shape,
        #                                     basePosition=[0, 0, 6.2],
        #                                     baseOrientation=[x, y, z, w],
        #                                     )  #
        cube_collision_shape = pbl.createCollisionShape(pbl.GEOM_BOX, halfExtents=[8, 0.02, 0.02])
        cube_visual_shape = pbl.createVisualShape(pbl.GEOM_BOX, halfExtents=[8, 0.02, 0.02],
                                                  rgbaColor=[14 / 255, 15 / 255, 17 / 255, 1])
        cube_body = pbl.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=cube_collision_shape,
                                        baseVisualShapeIndex=cube_visual_shape,
                                        basePosition=[0, 0, 6.2])

    def _housekeeping(self, percent):
        """Housekeeping function.

        Allocation and zero-ing of the variables and PyBullet's parameters/objects
        in the `reset()` function.

        """
        #### Initialize/reset counters and zero-valued variables ###
        self.RESET_TIME = time.time()
        self.step_counter = 0
        self.first_render_call = True
        self.X_AX = -1 * np.ones(self.NUM_DRONES)
        self.Y_AX = -1 * np.ones(self.NUM_DRONES)
        self.Z_AX = -1 * np.ones(self.NUM_DRONES)
        self.GUI_INPUT_TEXT = -1 * np.ones(self.NUM_DRONES)
        self.USE_GUI_RPM = False
        self.last_input_switch = 0
        self.last_clipped_action = np.zeros((self.NUM_DRONES, 4))
        self.gui_input = np.zeros(4)
        #### Initialize the drones kinemaatic information ##########
        self.pos = np.zeros((self.NUM_DRONES, 3))
        self.quat = np.zeros((self.NUM_DRONES, 4))
        self.rpy = np.zeros((self.NUM_DRONES, 3))
        self.vel = np.zeros((self.NUM_DRONES, 3))
        self.ang_v = np.zeros((self.NUM_DRONES, 3))
        if self.PHYSICS == Physics.DYN:
            self.rpy_rates = np.zeros((self.NUM_DRONES, 3))
        #### Set PyBullet's parameters #############################
        pbl.setGravity(0, 0, -self.G, physicsClientId=self.CLIENT)

        if SCENARIO == "Farm":
            self.hj_create_cylinder()

        pbl.setRealTimeSimulation(0, physicsClientId=self.CLIENT)
        pbl.setTimeStep(self.PYB_TIMESTEP, physicsClientId=self.CLIENT)
        pbl.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.CLIENT)
        #### Load ground plane, drone and obstacles models #########
        self.PLANE_ID = pbl.loadURDF("plane.urdf", physicsClientId=self.CLIENT)

        initial_xyzs = self.hj_random_init(percent)
        # print("initial_xyzs: ", initial_xyzs)
        self.INIT_XYZS = initial_xyzs

        self.DRONE_IDS = np.array(
            [pbl.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/' + self.URDF),
                          self.INIT_XYZS[i, :],
                          pbl.getQuaternionFromEuler(self.INIT_RPYS[i, :]),
                          flags=pbl.URDF_USE_INERTIA_FROM_FILE,
                          physicsClientId=self.CLIENT,
                          globalScaling=GLOBAL_SCALING,
                          ) for i in range(self.NUM_DRONES)])
        #### Remove default damping #################################
        # for i in range(self.NUM_DRONES):
        #     p.changeDynamics(self.DRONE_IDS[i], -1, linearDamping=0, angularDamping=0)
        #### Show the frame of reference of the drone, note that ###
        #### It severly slows down the GUI #########################
        if self.GUI and self.USER_DEBUG:
            for i in range(self.NUM_DRONES):
                self._showDroneLocalAxes(i)
        #### Disable collisions between drones' and the ground plane
        #### E.g., to start a drone at [0,0,0] #####################
        # for i in range(self.NUM_DRONES):
        # p.setCollisionFilterPair(bodyUniqueIdA=self.PLANE_ID, bodyUniqueIdB=self.DRONE_IDS[i], linkIndexA=-1, linkIndexB=-1, enableCollision=0, physicsClientId=self.CLIENT)
        if self.OBSTACLES:
            self._addObstacles(percent=percent)

    ################################################################################

    def _updateAndStoreKinematicInformation(self):
        """Updates and stores the drones kinemaatic information.

        This method is meant to limit the number of calls to PyBullet in each step
        and improve performance (at the expense of memory).

        """
        for i in range(self.NUM_DRONES):
            self.pos[i], self.quat[i] = pbl.getBasePositionAndOrientation(self.DRONE_IDS[i],
                                                                          physicsClientId=self.CLIENT)
            self.rpy[i] = pbl.getEulerFromQuaternion(self.quat[i])
            self.vel[i], self.ang_v[i] = pbl.getBaseVelocity(self.DRONE_IDS[i], physicsClientId=self.CLIENT)

    ################################################################################

    def _startVideoRecording(self):
        """Starts the recording of a video output.

        The format of the video output is .mp4, if GUI is True, or .png, otherwise.

        """
        if self.RECORD and self.GUI:
            self.VIDEO_ID = pbl.startStateLogging(loggingType=pbl.STATE_LOGGING_VIDEO_MP4,
                                                  fileName=os.path.join(self.OUTPUT_FOLDER,
                                                                        "video-" + datetime.now().strftime(
                                                                            "%m.%d.%Y_%H.%M.%S") + ".mp4"),
                                                  physicsClientId=self.CLIENT
                                                  )
        if self.RECORD and not self.GUI:
            self.FRAME_NUM = 0
            self.IMG_PATH = os.path.join(self.OUTPUT_FOLDER,
                                         "recording_" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"), '')
            os.makedirs(os.path.dirname(self.IMG_PATH), exist_ok=True)

    ################################################################################

    def _getDroneStateVector(self,
                             nth_drone
                             ):
        """Returns the state vector of the n-th drone.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        Returns
        -------
        ndarray 
            (20,)-shaped array of floats containing the state vector of the n-th drone.
            Check the only line in this method and `_updateAndStoreKinematicInformation()`
            to understand its format.

        """
        pos = self.pos[nth_drone, :]
        nth = self.quat[nth_drone, :]
        rpy = self.rpy[nth_drone, :]
        vel = self.vel[nth_drone, :]
        ang = self.ang_v[nth_drone, :]
        last_clipped_action = self.last_clipped_action[nth_drone, :]
        state = np.hstack([pos, nth, rpy, vel, ang, last_clipped_action])  # 3 4 3 3 3 4
        return state.reshape(20, )

    ################################################################################

    def _getDroneImages(self,
                        nth_drone,
                        segmentation: bool = True
                        ):
        """Returns camera captures from the n-th drone POV.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.
        segmentation : bool, optional
            Whehter to compute the compute the segmentation mask.
            It affects performance.

        Returns
        -------
        ndarray 
            (h, w, 4)-shaped array of uint8's containing the RBG(A) image captured from the n-th drone's POV.
        ndarray
            (h, w)-shaped array of uint8's containing the depth image captured from the n-th drone's POV.
        ndarray
            (h, w)-shaped array of uint8's containing the segmentation image captured from the n-th drone's POV.

        """
        if self.IMG_RES is None:
            print("[ERROR] in BaseAviary._getDroneImages(), remember to set self.IMG_RES to np.array([width, height])")
            exit()
        rot_mat = np.array(pbl.getMatrixFromQuaternion(self.quat[nth_drone, :])).reshape(3, 3)
        #### Set target point, camera view and projection matrices #
        target = np.dot(rot_mat, np.array([1000, 0, 0])) + np.array(self.pos[nth_drone, :])
        DRONE_CAM_VIEW = pbl.computeViewMatrix(cameraEyePosition=self.pos[nth_drone, :] + np.array([0, 0, self.L]),
                                               cameraTargetPosition=target,
                                               cameraUpVector=[0, 0, 1],
                                               physicsClientId=self.CLIENT
                                               )
        DRONE_CAM_PRO = pbl.computeProjectionMatrixFOV(fov=60.0,
                                                       aspect=1.0,
                                                       nearVal=self.L,
                                                       farVal=1000.0
                                                       )
        SEG_FLAG = pbl.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX if segmentation else pbl.ER_NO_SEGMENTATION_MASK
        [w, h, rgb, dep, seg] = pbl.getCameraImage(width=self.IMG_RES[0],
                                                   height=self.IMG_RES[1],
                                                   shadow=1,
                                                   viewMatrix=DRONE_CAM_VIEW,
                                                   projectionMatrix=DRONE_CAM_PRO,
                                                   flags=SEG_FLAG,
                                                   physicsClientId=self.CLIENT
                                                   )
        rgb = np.reshape(rgb, (h, w, 4))
        dep = np.reshape(dep, (h, w))
        seg = np.reshape(seg, (h, w))
        return rgb, dep, seg

    ################################################################################

    def _exportImage(self,
                     img_type: ImageType,
                     img_input,
                     path: str,
                     frame_num: int = 0
                     ):
        """Returns camera captures from the n-th drone POV.

        Parameters
        ----------
        img_type : ImageType
            The image type: RGB(A), depth, segmentation, or B&W (from RGB).
        img_input : ndarray
            (h, w, 4)-shaped array of uint8's for RBG(A) or B&W images.
            (h, w)-shaped array of uint8's for depth or segmentation images.
        path : str
            Path where to save the output as PNG.
        fram_num: int, optional
            Frame number to append to the PNG's filename.

        """
        if img_type == ImageType.RGB:
            (Image.fromarray(img_input.astype('uint8'), 'RGBA')).save(
                os.path.join(path, "frame_" + str(frame_num) + ".png"))
        elif img_type == ImageType.DEP:
            temp = ((img_input - np.min(img_input)) * 255 / (np.max(img_input) - np.min(img_input))).astype('uint8')
        elif img_type == ImageType.SEG:
            temp = ((img_input - np.min(img_input)) * 255 / (np.max(img_input) - np.min(img_input))).astype('uint8')
        elif img_type == ImageType.BW:
            temp = (np.sum(img_input[:, :, 0:2], axis=2) / 3).astype('uint8')
        else:
            print("[ERROR] in BaseAviary._exportImage(), unknown ImageType")
            exit()
        if img_type != ImageType.RGB:
            (Image.fromarray(temp)).save(os.path.join(path, "frame_" + str(frame_num) + ".png"))

    ################################################################################

    def _getAdjacencyMatrix(self):
        """Computes the adjacency matrix of a multi-drone system.

        Attribute NEIGHBOURHOOD_RADIUS is used to determine neighboring relationships.

        Returns
        -------
        ndarray
            (NUM_DRONES, NUM_DRONES)-shaped array of 0's and 1's representing the adjacency matrix 
            of the system: adj_mat[i,j] == 1 if (i, j) are neighbors; == 0 otherwise.

        """
        adjacency_mat = np.identity(self.NUM_DRONES)
        for i in range(self.NUM_DRONES - 1):
            for j in range(self.NUM_DRONES - i - 1):
                if np.linalg.norm(self.pos[i, :] - self.pos[j + i + 1, :]) < self.NEIGHBOURHOOD_RADIUS:
                    adjacency_mat[i, j + i + 1] = adjacency_mat[j + i + 1, i] = 1
        return adjacency_mat

    ################################################################################

    def _physics(self,
                 rpm,
                 nth_drone
                 ):
        """Base PyBullet physics implementation.

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        forces = np.array(rpm ** 2) * self.KF
        torques = np.array(rpm ** 2) * self.KM
        if self.DRONE_MODEL == DroneModel.RACE:
            torques = -torques
        z_torque = (-torques[0] + torques[1] - torques[2] + torques[3])
        for i in range(4):
            pbl.applyExternalForce(self.DRONE_IDS[nth_drone],
                                   i,
                                   forceObj=[0, 0, forces[i]],
                                   posObj=[0, 0, 0],
                                   flags=pbl.LINK_FRAME,
                                   physicsClientId=self.CLIENT
                                   )
        pbl.applyExternalTorque(self.DRONE_IDS[nth_drone],
                                4,
                                torqueObj=[0, 0, z_torque],
                                flags=pbl.LINK_FRAME,
                                physicsClientId=self.CLIENT
                                )

    ################################################################################

    def _groundEffect(self,
                      rpm,
                      nth_drone
                      ):
        """PyBullet implementation of a ground effect model.

        Inspired by the analytical model used for comparison in (Shi et al., 2019).

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        #### Kin. info of all links (propellers and center of mass)
        link_states = pbl.getLinkStates(self.DRONE_IDS[nth_drone],
                                        linkIndices=[0, 1, 2, 3, 4],
                                        computeLinkVelocity=1,
                                        computeForwardKinematics=1,
                                        physicsClientId=self.CLIENT
                                        )
        #### Simple, per-propeller ground effects ##################
        prop_heights = np.array(
            [link_states[0][0][2], link_states[1][0][2], link_states[2][0][2], link_states[3][0][2]])
        prop_heights = np.clip(prop_heights, self.GND_EFF_H_CLIP, np.inf)
        gnd_effects = np.array(rpm ** 2) * self.KF * self.GND_EFF_COEFF * (self.PROP_RADIUS / (4 * prop_heights)) ** 2
        if np.abs(self.rpy[nth_drone, 0]) < np.pi / 2 and np.abs(self.rpy[nth_drone, 1]) < np.pi / 2:
            for i in range(4):
                pbl.applyExternalForce(self.DRONE_IDS[nth_drone],
                                       i,
                                       forceObj=[0, 0, gnd_effects[i]],
                                       posObj=[0, 0, 0],
                                       flags=pbl.LINK_FRAME,
                                       physicsClientId=self.CLIENT
                                       )

    ################################################################################

    def _drag(self,
              rpm,
              nth_drone
              ):
        """PyBullet implementation of a drag model.

        Based on the the system identification in (Forster, 2015).

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        #### Rotation matrix of the base ###########################
        base_rot = np.array(pbl.getMatrixFromQuaternion(self.quat[nth_drone, :])).reshape(3, 3)
        #### Simple draft model applied to the base/center of mass #
        drag_factors = -1 * self.DRAG_COEFF * np.sum(np.array(2 * np.pi * rpm / 60))
        drag = np.dot(base_rot.T, drag_factors * np.array(self.vel[nth_drone, :]))
        pbl.applyExternalForce(self.DRONE_IDS[nth_drone],
                               4,
                               forceObj=drag,
                               posObj=[0, 0, 0],
                               flags=pbl.LINK_FRAME,
                               physicsClientId=self.CLIENT
                               )

    ################################################################################

    def _downwash(self,
                  nth_drone
                  ):
        """PyBullet implementation of a ground effect model.

        Based on experiments conducted at the Dynamic Systems Lab by SiQi Zhou.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        for i in range(self.NUM_DRONES):
            delta_z = self.pos[i, 2] - self.pos[nth_drone, 2]
            delta_xy = np.linalg.norm(np.array(self.pos[i, 0:2]) - np.array(self.pos[nth_drone, 0:2]))
            if delta_z > 0 and delta_xy < 10:  # Ignore drones more than 10 meters away
                alpha = self.DW_COEFF_1 * (self.PROP_RADIUS / (4 * delta_z)) ** 2
                beta = self.DW_COEFF_2 * delta_z + self.DW_COEFF_3
                downwash = [0, 0, -alpha * np.exp(-.5 * (delta_xy / beta) ** 2)]
                pbl.applyExternalForce(self.DRONE_IDS[nth_drone],
                                       4,
                                       forceObj=downwash,
                                       posObj=[0, 0, 0],
                                       flags=pbl.LINK_FRAME,
                                       physicsClientId=self.CLIENT
                                       )

    ################################################################################

    def _dynamics(self,
                  rpm,
                  nth_drone
                  ):
        """Explicit dynamics implementation.

        Based on code written at the Dynamic Systems Lab by James Xu.

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        #### Current state #########################################
        pos = self.pos[nth_drone, :]
        quat = self.quat[nth_drone, :]
        vel = self.vel[nth_drone, :]
        rpy_rates = self.rpy_rates[nth_drone, :]
        rotation = np.array(pbl.getMatrixFromQuaternion(quat)).reshape(3, 3)
        #### Compute forces and torques ############################
        forces = np.array(rpm ** 2) * self.KF
        thrust = np.array([0, 0, np.sum(forces)])
        thrust_world_frame = np.dot(rotation, thrust)
        force_world_frame = thrust_world_frame - np.array([0, 0, self.GRAVITY])
        z_torques = np.array(rpm ** 2) * self.KM
        if self.DRONE_MODEL == DroneModel.RACE:
            z_torques = -z_torques
        z_torque = (-z_torques[0] + z_torques[1] - z_torques[2] + z_torques[3])
        if self.DRONE_MODEL == DroneModel.CF2X or self.DRONE_MODEL == DroneModel.RACE:
            x_torque = (forces[0] + forces[1] - forces[2] - forces[3]) * (self.L / np.sqrt(2))
            y_torque = (- forces[0] + forces[1] + forces[2] - forces[3]) * (self.L / np.sqrt(2))
        elif self.DRONE_MODEL == DroneModel.CF2P:
            x_torque = (forces[1] - forces[3]) * self.L
            y_torque = (-forces[0] + forces[2]) * self.L
        torques = np.array([x_torque, y_torque, z_torque])
        torques = torques - np.cross(rpy_rates, np.dot(self.J, rpy_rates))
        rpy_rates_deriv = np.dot(self.J_INV, torques)
        no_pybullet_dyn_accs = force_world_frame / self.M
        #### Update state ##########################################
        vel = vel + self.PYB_TIMESTEP * no_pybullet_dyn_accs
        rpy_rates = rpy_rates + self.PYB_TIMESTEP * rpy_rates_deriv
        pos = pos + self.PYB_TIMESTEP * vel
        quat = self._integrateQ(quat, rpy_rates, self.PYB_TIMESTEP)
        #### Set PyBullet's state ##################################
        pbl.resetBasePositionAndOrientation(self.DRONE_IDS[nth_drone],
                                            pos,
                                            quat,
                                            physicsClientId=self.CLIENT
                                            )
        #### Note: the base's velocity only stored and not used ####
        pbl.resetBaseVelocity(self.DRONE_IDS[nth_drone],
                              vel,
                              np.dot(rotation, rpy_rates),
                              physicsClientId=self.CLIENT
                              )
        #### Store the roll, pitch, yaw rates for the next step ####
        self.rpy_rates[nth_drone, :] = rpy_rates

    def _integrateQ(self, quat, omega, dt):
        omega_norm = np.linalg.norm(omega)
        p, q, r = omega
        if np.isclose(omega_norm, 0):
            return quat
        lambda_ = np.array([
            [0, r, -q, p],
            [-r, 0, p, q],
            [q, -p, 0, r],
            [-p, -q, -r, 0]
        ]) * .5
        theta = omega_norm * dt / 2
        quat = np.dot(np.eye(4) * np.cos(theta) + 2 / omega_norm * lambda_ * np.sin(theta), quat)
        return quat

    ################################################################################

    def _normalizedActionToRPM(self,
                               action
                               ):
        """De-normalizes the [-1, 1] range to the [0, MAX_RPM] range.

        Parameters
        ----------
        action : ndarray
            (4)-shaped array of ints containing an input in the [-1, 1] range.

        Returns
        -------
        ndarray
            (4)-shaped array of ints containing RPMs for the 4 motors in the [0, MAX_RPM] range.

        """
        if np.any(np.abs(action) > 1):
            print("\n[ERROR] it", self.step_counter, "in BaseAviary._normalizedActionToRPM(), out-of-bound action")
        return np.where(action <= 0, (action + 1) * self.HOVER_RPM, self.HOVER_RPM + (
                self.MAX_RPM - self.HOVER_RPM) * action)  # Non-linear mapping: -1 -> 0, 0 -> HOVER_RPM, 1 -> MAX_RPM`

    ################################################################################

    def _showDroneLocalAxes(self,
                            nth_drone
                            ):
        """Draws the local frame of the n-th drone in PyBullet's GUI.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        if self.GUI:
            AXIS_LENGTH = 2 * self.L
            self.X_AX[nth_drone] = pbl.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                                        lineToXYZ=[AXIS_LENGTH, 0, 0],
                                                        lineColorRGB=[1, 0, 0],
                                                        parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                                                        parentLinkIndex=-1,
                                                        replaceItemUniqueId=int(self.X_AX[nth_drone]),
                                                        physicsClientId=self.CLIENT
                                                        )
            self.Y_AX[nth_drone] = pbl.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                                        lineToXYZ=[0, AXIS_LENGTH, 0],
                                                        lineColorRGB=[0, 1, 0],
                                                        parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                                                        parentLinkIndex=-1,
                                                        replaceItemUniqueId=int(self.Y_AX[nth_drone]),
                                                        physicsClientId=self.CLIENT
                                                        )
            self.Z_AX[nth_drone] = pbl.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                                        lineToXYZ=[0, 0, AXIS_LENGTH],
                                                        lineColorRGB=[0, 0, 1],
                                                        parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                                                        parentLinkIndex=-1,
                                                        replaceItemUniqueId=int(self.Z_AX[nth_drone]),
                                                        physicsClientId=self.CLIENT
                                                        )

    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment.

        These obstacles are loaded from standard URDF files included in Bullet.

        """
        pbl.loadURDF("samurai.urdf",
                     physicsClientId=self.CLIENT
                     )
        pbl.loadURDF("duck_vhacd.urdf",
                     [-.5, -.5, .05],
                     pbl.getQuaternionFromEuler([0, 0, 0]),
                     physicsClientId=self.CLIENT
                     )
        pbl.loadURDF("cube_no_rotation.urdf",
                     [-.5, -2.5, .5],
                     pbl.getQuaternionFromEuler([0, 0, 0]),
                     physicsClientId=self.CLIENT
                     )
        pbl.loadURDF("sphere2.urdf",
                     [0, 2, .5],
                     pbl.getQuaternionFromEuler([0, 0, 0]),
                     physicsClientId=self.CLIENT
                     )

    ################################################################################

    def _parseURDFParameters(self):
        """Loads parameters from an URDF file.

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.

        """
        URDF_TREE = etxml.parse(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/' + self.URDF)).getroot()
        M = float(URDF_TREE[1][0][1].attrib['value']) # * 15 #  * 15 * 15
        L = float(URDF_TREE[0].attrib['arm'])
        THRUST2WEIGHT_RATIO = float(URDF_TREE[0].attrib['thrust2weight'])
        IXX = float(URDF_TREE[1][0][2].attrib['ixx'])
        IYY = float(URDF_TREE[1][0][2].attrib['iyy'])
        IZZ = float(URDF_TREE[1][0][2].attrib['izz'])
        J = np.diag([IXX, IYY, IZZ])
        J_INV = np.linalg.inv(J)
        KF = float(URDF_TREE[0].attrib['kf'])
        KM = float(URDF_TREE[0].attrib['km'])
        COLLISION_H = float(URDF_TREE[1][2][1][0].attrib['length']) #* 15
        COLLISION_R = float(URDF_TREE[1][2][1][0].attrib['radius']) #* 15
        COLLISION_SHAPE_OFFSETS = [float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')]#  * 15
        COLLISION_Z_OFFSET = COLLISION_SHAPE_OFFSETS[2] #* 15
        MAX_SPEED_KMH = float(URDF_TREE[0].attrib['max_speed_kmh'])
        GND_EFF_COEFF = float(URDF_TREE[0].attrib['gnd_eff_coeff'])
        PROP_RADIUS = float(URDF_TREE[0].attrib['prop_radius'])
        DRAG_COEFF_XY = float(URDF_TREE[0].attrib['drag_coeff_xy'])
        DRAG_COEFF_Z = float(URDF_TREE[0].attrib['drag_coeff_z'])
        DRAG_COEFF = np.array([DRAG_COEFF_XY, DRAG_COEFF_XY, DRAG_COEFF_Z])
        DW_COEFF_1 = float(URDF_TREE[0].attrib['dw_coeff_1'])
        DW_COEFF_2 = float(URDF_TREE[0].attrib['dw_coeff_2'])
        DW_COEFF_3 = float(URDF_TREE[0].attrib['dw_coeff_3'])
        return M, L, THRUST2WEIGHT_RATIO, J, J_INV, KF, KM, COLLISION_H, COLLISION_R, COLLISION_Z_OFFSET, MAX_SPEED_KMH, \
            GND_EFF_COEFF, PROP_RADIUS, DRAG_COEFF, DW_COEFF_1, DW_COEFF_2, DW_COEFF_3

    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Must be implemented in a subclass.

        Parameters
        ----------
        action : ndarray | dict[..]
            The input action for one or more drones, to be translated into RPMs.

        """
        raise NotImplementedError

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def _computeTerminated(self):
        """Computes the current terminated value(s).

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def _computeTruncated(self):
        """Computes the current truncated value(s).

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def _calculateNextStep(self, current_position, destination, step_size=1):
        """
        Calculates intermediate waypoint
        towards drone's destination
        from drone's current position

        Enables drones to reach distant waypoints without
        losing control/crashing, and hover on arrival at destintion

        Parameters
        ----------
        current_position : ndarray
            drone's current position from state vector
        destination : ndarray
            drone's target position 
        step_size: int
            distance next waypoint is from current position, default 1

        Returns
        ----------
        next_pos: int 
            intermediate waypoint for drone

        """
        direction = (
                destination - current_position
        )  # Calculate the direction vector
        distance = np.linalg.norm(
            direction
        )  # Calculate the distance to the destination

        if distance <= step_size:
            # If the remaining distance is less than or equal to the step size,
            # return the destination
            return destination

        normalized_direction = (
                direction / distance
        )  # Normalize the direction vector
        next_step = (
                current_position + normalized_direction * step_size
        )  # Calculate the next step
        return next_step
