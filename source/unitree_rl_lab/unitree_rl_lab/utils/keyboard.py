# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).


"""Keyboard controller for SE(2) control."""

import weakref
from collections.abc import Callable

import carb
import omni
import torch
from isaaclab.devices.device_base import DeviceBase
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

class Keyboard(DeviceBase):

    def __init__(self, env: RslRlVecEnvWrapper):
        """Initialize the keyboard layer."""
        self.env = env
        # acquire omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        # note: Use weakref on callbacks to ensure that this object can be deleted when its destructor is called
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )
        # bindings for keyboard to command
        self._create_key_bindings()
        # dictionary for additional callbacks
        self._additional_callbacks = dict()
        ########################################
        self.x_vel = 0
        self.y_vel = 0
        self.yaw_vel = 0

    def __del__(self):
        """Release the keyboard interface."""
        self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = f"Keyboard Controller for ManagerBasedRLEnv: {self.__class__.__name__}\n"
        return msg

    """
    Operations
    """

    def reset(self):
        pass

    def add_callback(self, key: str, func: Callable):
        pass

    def advance(self):
        pass

    """
    Internal helpers.
    """

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Subscriber callback to when kit is updated.

        Reference:
            https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html
        """
        # apply the command when pressed
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._INPUT_KEY_MAPPING:
                if event.input.name == "R":
                    self.env.unwrapped.episode_length_buf = torch.ones_like(self.env.unwrapped.episode_length_buf) * 1e6
                if event.input.name == "W":
                    self.x_vel += 0.1
                    if "base_velocity" in self.env.unwrapped.command_manager._terms:
                        print("Command X Vel :%s" % str(self.env.unwrapped.command_manager.get_command("base_velocity")[:, 0]))
                if event.input.name == "S":
                    self.x_vel -= 0.1
                    if "base_velocity" in self.env.unwrapped.command_manager._terms:
                        print("Command X Vel :%s" % str(self.env.unwrapped.command_manager.get_command("base_velocity")[:, 0]))
                if event.input.name == "A":
                    self.y_vel += 0.1
                    if "base_velocity" in self.env.unwrapped.command_manager._terms:
                        print("Command Y Vel :%s" % str(self.env.unwrapped.command_manager.get_command("base_velocity")[:, 1]))
                if event.input.name == "D":
                    self.y_vel -= 0.1
                    if "base_velocity" in self.env.unwrapped.command_manager._terms:
                        print("Command Y Vel :%s" % str(self.env.unwrapped.command_manager.get_command("base_velocity")[:, 1]))
                if event.input.name == "Q":
                    self.yaw_vel += 0.1
                    if "base_velocity" in self.env.unwrapped.command_manager._terms:
                        print("Command Yaw Vel :%s" % str(self.env.unwrapped.command_manager.get_command("base_velocity")[:, 2]))
                if event.input.name == "E":
                    self.yaw_vel -= 0.1
                    if "base_velocity" in self.env.unwrapped.command_manager._terms:
                        print("Command Yaw Vel :%s" % str(self.env.unwrapped.command_manager.get_command("base_velocity")[:, 2]))
                if event.input.name == "X":
                    self.x_vel = 0
                    self.y_vel = 0
                    self.yaw_vel = 0
                    print("Reset All Command Vel")

        if "base_velocity" in self.env.unwrapped.command_manager._terms:
            self.env.unwrapped.command_manager.get_command("base_velocity")[:, 0] = self.x_vel
            self.env.unwrapped.command_manager.get_command("base_velocity")[:, 1] = self.y_vel
            self.env.unwrapped.command_manager.get_command("base_velocity")[:, 2] = self.yaw_vel

        # since no error, we are fine :)
        return True

    def _create_key_bindings(self):
        """Creates default key binding."""
        self._INPUT_KEY_MAPPING = {
            "R": "reset envs",
            "W": "forward command",
            "S": "backward command",
            "A": "leftward command",
            "D": "rightward command",
            "Q": "TernLeft command",
            "E": "TernRight command",
            "X": "Reset command",
        }
