#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down

    L            : toggle next light type
    SHIFT + L    : toggle high beam
    Z/X          : toggle right/left blinker
    I            : toggle interior light

    TAB          : change sensor position
    ` or N       : next sensor
    [1-9]        : change to sensor [1-9]
    G            : toggle radar visualization
    C            : change weather (Shift+C reverse)

    R            : toggle recording images to disk

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from examples.manual_control import (World, HUD, CameraManager,
                                     CollisionSensor, LaneInvasionSensor, GnssSensor, IMUSensor)

import os
import argparse
import logging
import time
import json
import socket
import traceback

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_g
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_l
    from pygame.locals import K_i
    from pygame.locals import K_z
    from pygame.locals import K_x
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================

class WorldSR(World):

    restarted = False

    def restart(self):

        if self.restarted:
            return
        self.restarted = True

        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713

        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0

        # Get the ego vehicle
        while self.player is None:
            print("Waiting for the ego vehicle...")
            time.sleep(1)
            possible_vehicles = self.world.get_actors().filter('vehicle.*')
            for vehicle in possible_vehicles:
                if vehicle.attributes['role_name'] == "hero":
                    print("Ego vehicle found")
                    self.player = vehicle
                    break
        
        self.player_name = self.player.type_id

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def tick(self, clock):
        if len(self.world.get_actors().filter(self.player_name)) < 1:
            return False

        self.hud.tick(self, clock)
        return True

# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================

class PlaybackControl(object):

    def __init__(self, world, playback):
        self._control_list = []
        self._index = 0
        self._world = world
        if isinstance(world.player, carla.Vehicle):
            self._lights = carla.VehicleLightState.NONE
            world.player.set_light_state(self._lights)

        records = None

        if playback:
            with open(playback) as fd:
                try:
                    records = json.load(fd)
                except json.JSONDecodeError:
                    pass

        if records and records['records']:

            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((socket.gethostname(), 3000))

            # Check if the simulation is correctly set up
            self.check_simulation_setup(records)

            # transform strs into VehicleControl commands
            for entry in records['records']:
                control = carla.VehicleControl(throttle=entry['control']['throttle'],
                                                steer=entry['control']['steer'],
                                                brake=entry['control']['brake'],
                                                hand_brake=entry['control']['hand_brake'],
                                                reverse=entry['control']['reverse'],
                                                manual_gear_shift=entry['control']['manual_gear_shift'],
                                                gear=entry['control']['gear'])
                self._control_list.append(control)

    def check_simulation_setup(self,records):

        vehicle_type_logd = records['vehicle']
        vehicle_type_used = self._world.player.type_id

        # Check the vehicle
        if vehicle_type_logd != vehicle_type_used:
            raise Exception("Logged vehicle type is {}, but {} is being used.".format(
                            vehicle_type_logd, vehicle_type_used))

        # Check the synchrony
        sim_settings = self._world.world.get_settings()
        sync = sim_settings.synchronous_mode
        if not sync:
            raise Exception("Simulation has to be set to synchronous")

        # Check the time step
        time_logd = records['time_step']
        time_used = sim_settings.fixed_delta_seconds
        if time_logd != time_used:
            raise Exception("Logged time_step ({} s) differs from the current one ({} s).".format(
                            time_logd, time_used))

    def parse_events(self, timestamp=None):


        if self._index < len(self._control_list):
            control = self._control_list[self._index]
            self._world.player.apply_control(control)
            self._index += 1

            if isinstance(control, carla.VehicleControl):
                current_lights = self._lights
                if control.brake:
                    current_lights |= carla.VehicleLightState.Brake
                else: # Remove the Brake flag
                    current_lights &= ~carla.VehicleLightState.Brake

                if control.reverse:
                    current_lights |= carla.VehicleLightState.Reverse
                else: # Remove the Reverse flag
                    current_lights &= ~carla.VehicleLightState.Reverse

                if current_lights != self._lights: # Change the light state only if necessary
                    self._lights = current_lights
                    self._world.player.set_light_state(carla.VehicleLightState(self._lights))

        else:
            print("JSON file has no more entries")

        try:
            self.socket.send(bytes("Done", "utf-8"))
        except BrokenPipeError:
            pass

class KeyboardControl(object):
    """
    Class that handles keyboard input.
    """

    def __init__(self, world, start_in_autopilot, clock, client, log=None):
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._lights = carla.VehicleLightState.NONE
            world.player.set_autopilot(self._autopilot_enabled)
            world.player.set_light_state(self._lights)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")

        self._steer_cache = 0.0
        self._world = world
        self._vehicle = world.player
        self._clock = clock
        self._client = client

        self._log = log
        self._log_data = {}

        if self._log:

            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((socket.gethostname(), 3000))

            if not world.world.get_settings().synchronous_mode:
                raise Exception("Simulation has to be synchronous")

            self._log_data.update({'records': []})
            self._log_data.update({'time_step': world.world.get_settings().fixed_delta_seconds})
            self._log_data.update({'vehicle': world.player_name})

        world.player.set_autopilot(self._autopilot_enabled)
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, timestamp=None):
        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights
        world = self._world
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    if self._autopilot_enabled:
                        world.player.set_autopilot(False)
                        world.restart()
                        world.player.set_autopilot(True)
                    else:
                        world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_g:
                    world.toggle_radar()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key == K_n:
                    world.camera_manager.next_sensor()
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if (world.recording_enabled):
                        self._client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        self._client.start_recorder("manual_recording.rec")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    # stop recorder
                    self._client.stop_recorder()
                    world.recording_enabled = False
                    # work around to fix camera at start of replaying
                    current_index = world.camera_manager.index
                    world.destroy_sensors()
                    # disable autopilot
                    self._autopilot_enabled = False
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification("Replaying file 'manual_recording.rec'")
                    # replayer
                    self._client.replay_file("manual_recording.rec", world.recording_start, 0, 0)
                    world.camera_manager.set_sensor(current_index)
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                                ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p and not pygame.key.get_mods() & KMOD_CTRL:
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification(
                            'Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_CTRL:
                        current_lights ^= carla.VehicleLightState.Special1
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_SHIFT:
                        current_lights ^= carla.VehicleLightState.HighBeam
                    elif event.key == K_l:
                        # Use 'L' key to switch between lights:
                        # closed -> position -> low beam -> fog
                        if not self._lights & carla.VehicleLightState.Position:
                            world.hud.notification("Position lights")
                            current_lights |= carla.VehicleLightState.Position
                        else:
                            world.hud.notification("Low beam lights")
                            current_lights |= carla.VehicleLightState.LowBeam
                        if self._lights & carla.VehicleLightState.LowBeam:
                            world.hud.notification("Fog lights")
                            current_lights |= carla.VehicleLightState.Fog
                        if self._lights & carla.VehicleLightState.Fog:
                            world.hud.notification("Lights off")
                            current_lights ^= carla.VehicleLightState.Position
                            current_lights ^= carla.VehicleLightState.LowBeam
                            current_lights ^= carla.VehicleLightState.Fog
                    elif event.key == K_i:
                        current_lights ^= carla.VehicleLightState.Interior
                    elif event.key == K_z:
                        current_lights ^= carla.VehicleLightState.LeftBlinker
                    elif event.key == K_x:
                        current_lights ^= carla.VehicleLightState.RightBlinker

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):

                if self._log:
                    self._parse_vehicle_keys(pygame.key.get_pressed(), timestamp.delta_seconds * 1000)
                else:
                    self._parse_vehicle_keys(pygame.key.get_pressed(), self._clock.get_time())
                self._control.reverse = self._control.gear < 0
                # Set automatic control-related vehicle lights
                if self._control.brake:
                    current_lights |= carla.VehicleLightState.Brake
                else: # Remove the Brake flag
                    current_lights &= ~carla.VehicleLightState.Brake
                if self._control.reverse:
                    current_lights |= carla.VehicleLightState.Reverse
                else: # Remove the Reverse flag
                    current_lights &= ~carla.VehicleLightState.Reverse
                if current_lights != self._lights: # Change the light state only if necessary
                    self._lights = current_lights
                    world.player.set_light_state(carla.VehicleLightState(self._lights))
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), self._clock.get_time(), world)

            world.player.apply_control(self._control)

            if self._log:
                self._record_control(timestamp)
                try:
                    self.socket.send(bytes("Done", "utf-8"))
                except BrokenPipeError:
                    pass


    def _record_control(self, timestamp):
        if self._log_data:
            transform = self._vehicle.get_transform()
            velocity = self._vehicle.get_velocity()
            angular_velocity = self._vehicle.get_angular_velocity()
            new_record = {'control': {'throttle': self._control.throttle,
                                        'steer': self._control.steer,
                                        'brake': self._control.brake,
                                        'hand_brake': self._control.hand_brake,
                                        'reverse': self._control.reverse,
                                        'manual_gear_shift': self._control.manual_gear_shift,
                                        'gear': self._control.gear
                                        },
                            'transform': {'x': transform.location.x,
                                        'y': transform.location.y,
                                        'z': transform.location.z,
                                        'roll': transform.rotation.roll,
                                        'pitch': transform.rotation.pitch,
                                        'yaw': transform.rotation.yaw
                                        },
                            'velocity': {'x': velocity.x,
                                        'y': velocity.y,
                                        'z': velocity.z
                                        },
                            'angular_velocity': {'x': angular_velocity.x,
                                                'y': angular_velocity.y,
                                                'z': angular_velocity.z
                                                }
                            }

            self._log_data['records'].append(new_record)

    def _parse_vehicle_keys(self, keys, milliseconds):
        if keys[K_UP] or keys[K_w]:
            self._control.throttle = min(self._control.throttle + 0.01, 1)
        else:
            self._control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            self._control.brake = min(self._control.brake + 0.2, 1)
        else:
            self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    def _parse_walker_keys(self, keys, milliseconds, world):
	        self._control.speed = 0.0
	        if keys[K_DOWN] or keys[K_s]:
	            self._control.speed = 0.0
	        if keys[K_LEFT] or keys[K_a]:
	            self._control.speed = .01
	            self._rotation.yaw -= 0.08 * milliseconds
	        if keys[K_RIGHT] or keys[K_d]:
	            self._control.speed = .01
	            self._rotation.yaw += 0.08 * milliseconds
	        if keys[K_UP] or keys[K_w]:
	            self._control.speed = world.player_max_speed_fast if pygame.key.get_mods() & KMOD_SHIFT else world.player_max_speed
	        self._control.jump = keys[K_SPACE]
	        self._rotation.yaw = round(self._rotation.yaw, 1)
	        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

    def __del__(self):
        # Get ready to log user commands
        if self._log and self._log_data:
            with open(self._log, 'w') as fd:
                json.dump(self._log_data, fd, indent=4, sort_keys=True)

# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================

def game_loop(args):

    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        sim_world = client.get_world()
        world = WorldSR(sim_world, hud, args)

        clock = pygame.time.Clock()

        # Setup the control of the ego vehicle
        checks_passed = False
        if args.playback:
            controller = PlaybackControl(world, args.playback)
        else:
            controller = KeyboardControl(world, args.autopilot, clock, client, args.log)

        # Main loop
        if args.log or args.playback:

            parser_int = sim_world.on_tick(controller.parse_events)
            sim_world.tick()  # Activates controller.parse_events once to avoid getting stuck
            checks_passed = True

            while True:
                clock.tick_busy_loop(60)
                if not world.tick(clock):
                    return
                world.render(display)
                pygame.display.flip()

        else:
            while True:
                clock.tick_busy_loop(60)
                if controller.parse_events():
                    return
                if not world.tick(clock):
                    return
                world.render(display)
                pygame.display.flip()

    except socket.timeout:
        print("The socket of the controller has timed out")
    except Exception as e:
        print(e)
        traceback.print_exc()

    finally:

        if checks_passed:
            sim_world.remove_on_tick(parser_int)
            controller.socket.close()

        if world is not None:
            world.destroy()

        pygame.quit()

# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================

def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='Enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '-l', '--log',
        type=str,
        help='Filename to log user actions')
    argparser.add_argument(
        '--playback',
        type=str,
        help='Log filename to replay user actions')
    args = argparser.parse_args()


    args.rolename = 'hero'      # Needed for CARLA version
    args.filter = "vehicle.*"   # Needed for CARLA version
    args.gamma = 2.2   # Needed for CARLA version
    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    except Exception as error:
        logging.exception(error)

if __name__ == '__main__':
    main()
