from math import pi

import pygame
import pymunk

from constants import CENTER, RAGDOLL_GROUP
from utils import convert_coord, linear_conv


class BoneJoint:
    def __init__(self, space, x, y, mass, radius=6, group=RAGDOLL_GROUP):
        self.body = pymunk.Body(mass, pymunk.moment_for_circle(mass, 0, radius))
        self.body.position = x, y

        self.radius = radius

        self.shape = pymunk.Circle(self.body, radius=self.radius)
        self.shape.elasticity = 0.5
        self.shape.density = 1
        self.shape.friction = 0.4
        self.shape.filter = pymunk.ShapeFilter(group=group)

        space.add(self.shape, self.body)

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 0, 0), convert_coord(self.body.position), self.radius)


class Bone:
    def __init__(self, space, body1, body2, min_rot, max_rot, strength=10000):
        self.body1 = body1
        self.body2 = body2
        self.min_rot = min_rot
        self.max_rot = max_rot
        pin_joint = pymunk.PinJoint(self.body1, self.body2, (0, 0), (0, 0))
        rot_joint = pymunk.RotaryLimitJoint(self.body1, self.body2, self.min_rot, self.max_rot)
        space.add(pin_joint, rot_joint)

        self.strength = strength

    def draw(self, screen):
        pos1 = self.body1.position
        pos2 = self.body2.position
        pygame.draw.line(screen, (0, 0, 0), convert_coord(pos1), convert_coord(pos2), 5)

    def move(self, force):
        direction = (self.body2.position - self.body1.position).normalized()
        self.body2.apply_force_at_world_point(
            max(min(force, self.strength), -self.strength) * direction, self.body2.position
        )


class Ragdoll:
    def __init__(self, space, x, y):
        self.starting_positions = {
            "shoulders": (x, y + 80),
            "head": (x, y + 115),
            "left_elbow": (x - 40, y + 80),
            "left_hand": (x - 85, y + 80),
            "right_elbow": (x + 40, y + 80),
            "right_hand": (x + 85, y + 80),
            "hip": (x, y),
            "left_knee": (x - 25, y - 70),
            "right_knee": (x + 25, y - 70),
            "left_foot": (x - 40, y - 120),
            "right_foot": (x + 40, y - 120),
        }

        # Define mass for each joint
        joint_masses = {
            "shoulders": 15,
            "head": 10,
            "left_elbow": 8,
            "left_hand": 5,
            "right_elbow": 8,
            "right_hand": 5,
            "hip": 35,
            "left_knee": 18,
            "right_knee": 18,
            "left_foot": 10,
            "right_foot": 10,
        }

        self.joints = {
            name: BoneJoint(space, pos[0], pos[1], joint_masses[name]) for name, pos in self.starting_positions.items()
        }

        self.bones = {
            "neck": Bone(space, self.joints["shoulders"].body, self.joints["head"].body, -pi * 0.25, pi * 0.25),
            "left_arm": Bone(space, self.joints["shoulders"].body, self.joints["left_elbow"].body, -pi * 0.5, pi * 0.5),
            "left_forearm": Bone(space, self.joints["left_elbow"].body, self.joints["left_hand"].body, 0, pi),
            "right_arm": Bone(
                space, self.joints["shoulders"].body, self.joints["right_elbow"].body, -pi * 0.5, pi * 0.5
            ),
            "right_forearm": Bone(space, self.joints["right_elbow"].body, self.joints["right_hand"].body, 0, pi),
            "left_thigh": Bone(space, self.joints["hip"].body, self.joints["left_knee"].body, -pi * 0.25, pi),
            "left_leg": Bone(space, self.joints["left_knee"].body, self.joints["left_foot"].body, -pi * 0.5, 0),
            "right_thigh": Bone(space, self.joints["hip"].body, self.joints["right_knee"].body, -pi * 0.25, pi * 0.5),
            "right_leg": Bone(space, self.joints["right_knee"].body, self.joints["right_foot"].body, -pi * 0.5, 0),
            "torso": Bone(space, self.joints["hip"].body, self.joints["shoulders"].body, -pi * 0.5, pi * 0.5),
        }

    def draw(self, screen):
        for bone in self.bones.values():
            bone.draw(screen)

        for joint in self.joints.values():
            joint.draw(screen)

    def get_data(self):
        values = []
        hip_pos = self.joints["hip"].body.position
        for joint in self.joints.values():
            values.extend(joint.body.position - hip_pos)
        return values

    def move(self, forces):
        for force, bone in zip(forces, self.bones.values()):
            bone.move(linear_conv(force, -1, 1, -bone.strength, bone.strength))

    def calculate_fitness(self):
        fitness = 0

        # current = self.get_data()

        # Calculate how close each joint is to its starting position
        hip_pos = self.joints["hip"].body.position
        for i, joint_name in enumerate(self.joints):
            start_pos = self.starting_positions[joint_name]
            current_relative_pos = self.joints[joint_name].body.position - hip_pos
            distance = (current_relative_pos - start_pos).length
            fitness -= distance
        fitness *= 5 / len(self.joints)

        # Calculate distance of hips to center of the screen
        hips_distance = (hip_pos - CENTER).length
        fitness -= hips_distance

        return fitness
