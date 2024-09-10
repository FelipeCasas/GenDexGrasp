import json
import os
from tabnanny import check

import pickle
import matplotlib.pyplot as plt
import numpy as np
import pytorch_kinematics as pk
import torch
import torch.nn
import transforms3d
import trimesh as tm
import urdf_parser_py.urdf as URDF_PARSER
from plotly import graph_objects as go
from pytorch_kinematics.urdf_parser_py.urdf import (URDF, Box, Cylinder, Mesh, Sphere)
from utils.rot6d import *
from utils.utils_math import *
import trimesh.sample
# from kaolin.metrics.trianglemesh import point_to_mesh_distance
# from kaolin.ops.mesh import check_sign, index_vertices_by_faces, face_normals

from utils.visualize_plotly import plot_mesh


class XHandModel:
    def __init__(self, robot_name, urdf_filename, mesh_path,
                 batch_size=1, 
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 mesh_nsp=128,
                 hand_scale=2.
                 ):
        self.device = device
        self.robot_name = robot_name
        self.batch_size = batch_size
        # prepare model
        self.robot = pk.build_chain_from_urdf(open(urdf_filename).read()).to(dtype=torch.float, device=self.device)
        self.robot_full = URDF_PARSER.URDF.from_xml_file(urdf_filename)
        # prepare contact point basis and surface point samples
        # self.no_contact_dict = json.load(open(os.path.join('data', 'urdf', 'intersection_%s.json'%robot_name)))

        # prepare geometries for visualization
        self.global_translation = None
        self.global_rotation = None
        self.softmax = torch.nn.Softmax(dim=-1)
        # prepare contact point basis and surface point samples
        self.contact_point_dict = json.load(open(os.path.join("data/urdf/", 'contact_%s.json' % robot_name)))
        self.contact_point_basis = {}
        self.contact_normals = {}
        self.surface_points = {}
        self.surface_points_normal = {}
        visual = URDF.from_xml_string(open(urdf_filename).read())
        self.mesh_verts = {}
        self.mesh_faces = {}
        
        self.canon_verts = []
        self.canon_faces = []
        self.idx_vert_faces = []
        self.face_normals = []
        verts_bias = 0
            
        for i_link, link in enumerate(visual.links):
            print(f"Processing link #{i_link}: {link.name}")
            # load mesh
            if len(link.visuals) == 0:
                continue
            if type(link.visuals[0].geometry) == Mesh:
                # print(link.visuals[0])
                if robot_name == 'shadowhand' or robot_name == 'allegro' or robot_name == 'barrett':
                    filename = link.visuals[0].geometry.filename.split('/')[-1]
                elif robot_name == 'allegro':
                    filename = f"{link.visuals[0].geometry.filename.split('/')[-2]}/{link.visuals[0].geometry.filename.split('/')[-1]}"
                else:
                    filename = link.visuals[0].geometry.filename
                mesh = tm.load(os.path.join(mesh_path, filename), force='mesh', process=False)
            elif type(link.visuals[0].geometry) == Cylinder:
                mesh = tm.primitives.Cylinder(
                    radius=link.visuals[0].geometry.radius, height=link.visuals[0].geometry.length)
            elif type(link.visuals[0].geometry) == Box:
                mesh = tm.primitives.Box(extents=link.visuals[0].geometry.size)
            elif type(link.visuals[0].geometry) == Sphere:
                mesh = tm.primitives.Sphere(
                    radius=link.visuals[0].geometry.radius)
            else:
                print(type(link.visuals[0].geometry))
                raise NotImplementedError
            try:
                scale = np.array(
                    link.visuals[0].geometry.scale).reshape([1, 3])
            except:
                scale = np.array([[1, 1, 1]])
            try:
                rotation = transforms3d.euler.euler2mat(*link.visuals[0].origin.rpy)
                translation = np.reshape(link.visuals[0].origin.xyz, [1, 3])
                # print('---')
                # print(link.visuals[0].origin.rpy, rotation)
                # print('---')
            except AttributeError:
                rotation = transforms3d.euler.euler2mat(0, 0, 0)
                translation = np.array([[0, 0, 0]])
                
            # Surface point
            # mesh.sample(int(mesh.area * 100000)) * scale
            # todo: marked original count is 128
            if self.robot_name == 'shadowhand':
                pts, pts_face_index = trimesh.sample.sample_surface(mesh=mesh, count=64)
                pts_normal = np.array([mesh.face_normals[x] for x in pts_face_index], dtype=float)
            else:
                pts, pts_face_index = trimesh.sample.sample_surface(mesh=mesh, count=128)
                pts_normal = np.array([mesh.face_normals[x] for x in pts_face_index], dtype=float)

            if self.robot_name == 'barrett':
                if link.name in ['bh_base_link']:
                    pts = trimesh.sample.volume_mesh(mesh=mesh, count=1024)
                    pts_normal = np.array([[0., 0., 1.] for x in range(pts.shape[0])], dtype=float)
            if self.robot_name == 'ezgripper':
                if link.name in ['left_ezgripper_palm_link']:
                    pts = trimesh.sample.volume_mesh(mesh=mesh, count=1024)
                    pts_normal = np.array([[1., 0., 0.] for x in range(pts.shape[0])], dtype=float)
            if self.robot_name == 'robotiq_3finger':
                if link.name in ['gripper_palm']:
                    pts = trimesh.sample.volume_mesh(mesh=mesh, count=1024)
                    pts_normal = np.array([[0., 0., 1.] for x in range(pts.shape[0])], dtype=float)

            pts *= scale
            # pts = mesh.sample(128) * scale
            # print(link.name, len(pts))
            # new
            if robot_name == 'shadowhand':
                pts = pts[:, [0, 2, 1]]
                pts_normal = pts_normal[:, [0, 2, 1]]
                pts[:, 1] *= -1
                pts_normal[:, 1] *= -1

            pts = np.matmul(rotation, pts.T).T + translation
            # pts_normal = np.matmul(rotation, pts_normal.T).T
            pts = np.concatenate([pts, np.ones([len(pts), 1])], axis=-1)
            pts_normal = np.concatenate([pts_normal, np.ones([len(pts_normal), 1])], axis=-1)
            self.surface_points[link.name] = torch.from_numpy(pts).to(
                device).float().unsqueeze(0).repeat(batch_size, 1, 1)
            self.surface_points_normal[link.name] = torch.from_numpy(pts_normal).to(
                device).float().unsqueeze(0).repeat(batch_size, 1, 1)

            # visualization mesh
            self.mesh_verts[link.name] = np.array(mesh.vertices) * scale
            if robot_name == 'shadowhand':
                self.mesh_verts[link.name] = self.mesh_verts[link.name][:, [0, 2, 1]]
                self.mesh_verts[link.name][:, 1] *= -1
            self.mesh_verts[link.name] = np.matmul(rotation, self.mesh_verts[link.name].T).T + translation
            self.mesh_faces[link.name] = np.array(mesh.faces)

            # point and normal of palm center

            # contact point
            if link.name in self.contact_point_dict:
                # if link.name != 'index_1': continue
                # new 1.11
                cpb = np.array(self.contact_point_dict[link.name])
                # print("cpb shape: ", cpb.shape, len(cpb.shape))
                if len(cpb.shape) > 1:
                    cpb = cpb[np.random.randint(cpb.shape[0], size=1)][0]
                # print(link.name, cpb)
                # go.Figure(data = [
                #     go.Mesh3d(x=mesh.vertices[:,0], y=mesh.vertices[:,1], z=mesh.vertices[:,2], i=mesh.faces[:,0], j=mesh.faces[:,1], k=mesh.faces[:,2]),
                #     go.Scatter3d(x=mesh.vertices[cpb,0], y=mesh.vertices[cpb, 1], z=mesh.vertices[cpb,2])]).show()
                # input()

                cp_basis = mesh.vertices[cpb] * scale
                # print(cpb, "cp_basis: ", cp_basis)
                if robot_name == 'shadowhand':
                    cp_basis = cp_basis[:, [0, 2, 1]]
                    cp_basis[:, 1] *= -1
                cp_basis = np.matmul(rotation, cp_basis.T).T + translation
                cp_basis = torch.cat([torch.from_numpy(cp_basis).to(device).float(), torch.ones([4, 1]).to(device).float()], dim=-1)
                self.contact_point_basis[link.name] = cp_basis.unsqueeze( 0).repeat(batch_size, 1, 1)
                v1 = cp_basis[1, :3] - cp_basis[0, :3]
                v2 = cp_basis[2, :3] - cp_basis[0, :3]
                v1 = v1 / torch.norm(v1)
                v2 = v2 / torch.norm(v2)
                self.contact_normals[link.name] = torch.cross(v1, v2).view([1, 3])
                self.contact_normals[link.name] = self.contact_normals[link.name].unsqueeze(0).repeat(batch_size, 1, 1)
                
            # # Canonical hand meshes for penetration computation
            # self.canon_verts.append(torch.tensor(self.mesh_verts[link.name]).to(device).float().unsqueeze(0) * hand_scale)
            # self.canon_faces.append(torch.Tensor(mesh.faces).long().to(self.device))
            # self.idx_vert_faces.append(index_vertices_by_faces(self.canon_verts[-1], self.canon_faces[-1]))
            # self.face_normals.append(face_normals(self.idx_vert_faces[-1], unit=True))
            
        # new 2.1
        self.revolute_joints = []
        for i in range(len(self.robot_full.joints)):
            if self.robot_full.joints[i].joint_type == 'revolute':
                self.revolute_joints.append(self.robot_full.joints[i])
        self.revolute_joints_q_mid = []
        self.revolute_joints_q_var = []
        self.revolute_joints_q_upper = []
        self.revolute_joints_q_lower = []
        for i in range(len(self.robot.get_joint_parameter_names())):
            for j in range(len(self.revolute_joints)):
                if self.revolute_joints[j].name == self.robot.get_joint_parameter_names()[i]:
                    joint = self.revolute_joints[j]
            assert joint.name == self.robot.get_joint_parameter_names()[i]
            self.revolute_joints_q_mid.append(
                (joint.limit.lower + joint.limit.upper) / 2)
            self.revolute_joints_q_var.append(
                ((joint.limit.upper - joint.limit.lower) / 2) ** 2)
            self.revolute_joints_q_lower.append(joint.limit.lower)
            self.revolute_joints_q_upper.append(joint.limit.upper)

        self.revolute_joints_q_lower = torch.Tensor(
            self.revolute_joints_q_lower).repeat([self.batch_size, 1]).to(device)
        self.revolute_joints_q_upper = torch.Tensor(
            self.revolute_joints_q_upper).repeat([self.batch_size, 1]).to(device)

        self.current_status = None

        self.scale = hand_scale
        

    def update_kinematics(self, q):
        self.global_translation = q[:, :3]

        self.global_rotation = robust_compute_rotation_matrix_from_ortho6d(q[:,3:9])
        self.current_status = self.robot.forward_kinematics(q[:,9:])
        
    # def penetration(self, obj_pts, q=None, debug=False):
    #     """Penetration of object points in the conanical hand frame

    #     Args:
    #         q: B x l
    #         obj_pts: B x N x 4
    #     """
    #     if q is not None:
    #         self.update_kinematics(q)
    #     oh_pen = torch.zeros([1, obj_pts.shape[0] * obj_pts.shape[1]], device=self.device)

    #     # Transform point to the hand frame
    #     local_obj_pts = torch.matmul(self.global_rotation.transpose(1, 2), (obj_pts[..., :3] - self.global_translation.unsqueeze(1) * self.scale).transpose(1, 2)).transpose(1, 2)
    #     pts_shape = obj_pts[..., :3].shape
        
    #     for link_idx, link_name in enumerate(self.surface_points):
    #         # Transform point to the canonical part frame
    #         trans_matrix = self.current_status[link_name].get_matrix()
    #         lp_obj_pts = (torch.matmul(trans_matrix[:, :3, :3].transpose(1, 2), (local_obj_pts.clone() - trans_matrix[:, :3, -1].unsqueeze(1) * self.scale).transpose(1, 2)).transpose(1, 2))
    #         _lp_obj_pts = lp_obj_pts.contiguous().reshape((1, -1, 3))
    #         # Compute penetration
    #         oh_dist, _, _ = point_to_mesh_distance(_lp_obj_pts, self.idx_vert_faces[link_idx])
    #         oh_sign = check_sign(self.canon_verts[link_idx], self.canon_faces[link_idx], _lp_obj_pts)
    #         oh_pen = oh_pen + torch.where(oh_sign, oh_dist, torch.zeros_like(oh_dist, device=self.device))
            
    #         # if debug:
    #         #     from utils.visualize_plotly import plot_point_cloud
                
    #         #     go.Figure([ 
    #         #             # plot_point_cloud(self.canon_verts[link_idx][0].detach().cpu(), color='lightpink'),
    #         #             plot_point_cloud(lp_obj_pts[0].detach().cpu(), color='lightblue'),
    #         #             plot_mesh(tm.Trimesh(self.canon_verts[link_idx][0].detach().cpu(), self.mesh_faces[link_name], color='lightblue'))
    #         #     ]).show()
    #         #     input()
    #     return oh_pen.reshape((pts_shape[0], pts_shape[1]))
            
    def get_contact_points(self, contact_point_part_indices, contact_point_weights, q=None):
        contact_point_weights = self.softmax(contact_point_weights)
        if q is not None:
            self.update_kinematics(q)
        contact_point_basis_transformed = []
        # step 1: transform contact point basis
        for link_name in self.contact_point_basis:
            # get transformation
            trans_matrix = self.current_status[link_name].get_matrix()
            # get contact point
            cp_basis = self.contact_point_basis[link_name]
            contact_normal_orig = self.contact_normals[link_name]
            # cp_basis: B x 4 x 3
            cp_basis_transformed = torch.matmul(trans_matrix, torch.transpose(cp_basis, 1, 2)).transpose(1, 2)[..., :3]
            contact_point_basis_transformed.append(cp_basis_transformed)    # B x 4 x 3
        contact_point_basis_transformed = torch.stack(
            contact_point_basis_transformed, 1)  # B x J x 4 x 3
        # step 2: collect contact point basis corresponding to each contact point
        contact_point_basis_transformed = contact_point_basis_transformed[torch.arange(0, len(contact_point_basis_transformed), device=self.device).unsqueeze(1).long(), contact_point_part_indices.long()]
        # step 3: compute contact point coordinates
        contact_point_basis_transformed = (contact_point_basis_transformed * contact_point_weights.unsqueeze(-1)).sum(2)
        contact_point_basis_transformed = torch.matmul(self.global_rotation, contact_point_basis_transformed.transpose(1, 2)).transpose(1, 2) + self.global_translation.unsqueeze(1)
        return contact_point_basis_transformed * self.scale

    def get_contact_points_and_normal(self, contact_point_part_indices, contact_point_weights, q=None):
        contact_point_weights = self.softmax(contact_point_weights)
        if q is not None:
            self.update_kinematics(q)
        contact_point_basis_transformed = []
        contact_normal_transformed = []
        # step 1: transform contact point basis
        for link_name in self.contact_point_basis:
            # get transformation
            trans_matrix = self.current_status[link_name].get_matrix()
            # get contact point
            cp_basis = self.contact_point_basis[link_name]
            contact_normal_orig = self.contact_normals[link_name]
            # cp_basis: B x 4 x 3
            cp_basis_transformed = torch.matmul(
                trans_matrix, cp_basis.transpose(1, 2)).transpose(1, 2)[..., :3]
            contact_point_basis_transformed.append(
                cp_basis_transformed)    # B x 4 x 3
            contact_normal_transformed.append(torch.matmul(trans_matrix[..., :3, :3], torch.transpose(contact_normal_orig, 1, 2)).transpose(1, 2))  # B x 1 x 3
        contact_point_basis_transformed = torch.stack(
            contact_point_basis_transformed, 1)  # B x J x 4 x 3
        contact_normal_transformed = torch.stack(
            contact_normal_transformed, 1)  # B x J x 1 x 3
        # step 2: collect contact point basis corresponding to each contact point
        contact_point_basis_transformed = contact_point_basis_transformed[
            torch.arange(0, len(contact_point_basis_transformed), device=self.device).unsqueeze(1).long(), contact_point_part_indices.long()]
        contact_normal_transformed = contact_normal_transformed[
            torch.arange(0, len(contact_normal_transformed), device=self.device).unsqueeze(1).long(), contact_point_part_indices.long()].squeeze(2)  # B x J x 3
        # # step 3: compute contact point coordinates
        contact_point_basis_transformed = (
            contact_point_basis_transformed * contact_point_weights.unsqueeze(-1)).sum(2)
        contact_point_basis_transformed = torch.matmul(self.global_rotation, contact_point_basis_transformed.transpose(
            1, 2)).transpose(1, 2) + self.global_translation.unsqueeze(1)
        contact_normal_transformed = torch.matmul(
            self.global_rotation, contact_normal_transformed.transpose(1, 2)).transpose(1, 2)
        return contact_point_basis_transformed * self.scale, contact_normal_transformed

    def get_surface_points_prior(self, q=None, downsample=True):
        if q is not None:
            self.update_kinematics(q)
        surface_points = []

        for link_name in self.surface_points:
        # for link_name in parts:
            # get transformation
            trans_matrix = self.current_status[link_name].get_matrix()
            surface_points.append(torch.matmul(trans_matrix, self.surface_points[link_name].transpose(1, 2)).transpose(1, 2)[..., :3])
        surface_points = torch.cat(surface_points, 1)
        surface_points = torch.matmul(self.global_rotation, surface_points.transpose(1, 2)).transpose(1, 2) + self.global_translation.unsqueeze(1)
        # if downsample:
        #     surface_points = surface_points[:, torch.randperm(surface_points.shape[1])][:, :778]
        return surface_points * self.scale

    def get_surface_points(self, q=None, downsample=True):
        if q is not None:
            self.update_kinematics(q)
        surface_points = []

        for link_name in self.surface_points:
        # for link_name in parts:
            # get transformation
            trans_matrix = self.current_status[link_name].get_matrix()
            surface_points.append(torch.matmul(trans_matrix, self.surface_points[link_name].transpose(1, 2)).transpose(1, 2)[..., :3])
        surface_points = torch.cat(surface_points, 1)
        surface_points = torch.matmul(self.global_rotation, surface_points.transpose(1, 2)).transpose(1, 2) + self.global_translation.unsqueeze(1)
        # if downsample:
        #     surface_points = surface_points[:, torch.randperm(surface_points.shape[1])][:, :778]
        return surface_points * self.scale

    def get_surface_points_new(self, q=None, downsample=True):
        if q is not None:
            self.update_kinematics(q)
        surface_points = []
        for link_name in self.surface_points:
            if self.robot_name == 'robotiq_3finger' and link_name == 'gripper_palm':
                continue
            if self.robot_name == 'robotiq_3finger_real_robot' and link_name == 'palm':
                continue
            trans_matrix = self.current_status[link_name].get_matrix()
            surface_points.append(torch.matmul(trans_matrix, self.surface_points[link_name].transpose(1, 2)).transpose(1, 2)[..., :3])
        surface_points = torch.cat(surface_points, 1)
        surface_points = torch.matmul(self.global_rotation, surface_points.transpose(1, 2)).transpose(1, 2) + self.global_translation.unsqueeze(1)
        # if downsample:
        #     surface_points = surface_points[:, torch.randperm(surface_points.shape[1])][:, :778]
        return surface_points * self.scale

    def get_surface_points_paml(self, q=None):
        if q is not None:
            self.update_kinematics(q)
        surface_points = []
        if self.robot_name == 'allegro':
            palm_list = ['base_link']
        elif self.robot_name == 'robotiq_3finger_real_robot':
            palm_list = ['palm']
        else:
            raise NotImplementedError
        for link_name in palm_list:
        # for link_name in parts:
            # get transformation
            trans_matrix = self.current_status[link_name].get_matrix()
            surface_points.append(torch.matmul(trans_matrix, self.surface_points[link_name].transpose(1, 2)).transpose(1, 2)[..., :3])
        surface_points = torch.cat(surface_points, 1)
        surface_points = torch.matmul(self.global_rotation, surface_points.transpose(1, 2)).transpose(1, 2) + self.global_translation.unsqueeze(1)
        # if downsample:
        #     surface_points = surface_points[:, torch.randperm(surface_points.shape[1])][:, :778]
        return surface_points * self.scale

    def get_surface_points_and_normals(self, q=None):
        if q is not None:
            self.update_kinematics(q=q)
        surface_points = []
        surface_normals = []

        for link_name in self.surface_points:
            # for link_name in parts:
            # get transformation
            trans_matrix = self.current_status[link_name].get_matrix()
            surface_points.append(torch.matmul(trans_matrix, self.surface_points[link_name].transpose(1, 2)).transpose(1, 2)[..., :3])
            surface_normals.append(torch.matmul(trans_matrix, self.surface_points_normal[link_name].transpose(1, 2)).transpose(1, 2)[..., :3])
        surface_points = torch.cat(surface_points, 1)
        surface_normals = torch.cat(surface_normals, 1)
        surface_points = torch.matmul(self.global_rotation, surface_points.transpose(1, 2)).transpose(1, 2) + self.global_translation.unsqueeze(1)
        surface_normals = torch.matmul(self.global_rotation, surface_normals.transpose(1, 2)).transpose(1, 2)

        return surface_points * self.scale, surface_normals

    def get_key_points(self, links, q=None):
        if q is not None:
            self.update_kinematics(q)
        key_points = []
        for link_name in links:
            # get transformation
            trans_matrix = self.current_status[link_name].get_matrix()
            key_points.append(torch.matmul(
                trans_matrix,
                self.surface_points[link_name].mean(dim=1, keepdim=True).transpose(1, 2)).transpose(1, 2)[..., :3])
        key_points = torch.cat(key_points, 1)
        key_points = torch.matmul(self.global_rotation, key_points.transpose(1, 2)).transpose(1, 2) + self.global_translation.unsqueeze(1)
        return (key_points * self.scale).squeeze(0)

    def get_meshes_from_q(self, q=None, i=0):
        data = []
        if q is not None: self.update_kinematics(q)
        for idx, link_name in enumerate(self.mesh_verts):
            trans_matrix = self.current_status[link_name].get_matrix()
            trans_matrix = trans_matrix[min(len(trans_matrix) - 1, i)].detach().cpu().numpy()
            v = self.mesh_verts[link_name]
            transformed_v = np.concatenate([v, np.ones([len(v), 1])], axis=-1)
            transformed_v = np.matmul(trans_matrix, transformed_v.T).T[..., :3]
            transformed_v = np.matmul(self.global_rotation[i].detach().cpu().numpy(),
                                      transformed_v.T).T + np.expand_dims(
                self.global_translation[i].detach().cpu().numpy(), 0)
            transformed_v = transformed_v * self.scale
            f = self.mesh_faces[link_name]
            data.append(tm.Trimesh(vertices=transformed_v, faces=f))
        return data
    
    def get_plotly_data(self, q=None, i=0, color='lightblue', opacity=1.):
        data = []
        if q is not None: self.update_kinematics(q)
        for idx, link_name in enumerate(self.mesh_verts):
            trans_matrix = self.current_status[link_name].get_matrix()
            trans_matrix = trans_matrix[min(len(trans_matrix) - 1, i)].detach().cpu().numpy()
            v = self.mesh_verts[link_name]
            transformed_v = np.concatenate([v, np.ones([len(v), 1])], axis=-1)
            transformed_v = np.matmul(trans_matrix, transformed_v.T).T[..., :3]
            transformed_v = np.matmul(self.global_rotation[i].detach().cpu().numpy(),
                                      transformed_v.T).T + np.expand_dims(
                self.global_translation[i].detach().cpu().numpy(), 0)
            transformed_v = transformed_v * self.scale
            f = self.mesh_faces[link_name]
            data.append(
                go.Mesh3d(x=transformed_v[:, 0], y=transformed_v[:, 1], z=transformed_v[:, 2], i=f[:, 0], j=f[:, 1],
                          k=f[:, 2], color=color, opacity=opacity))
        return data


class HandModel:
    def __init__(
        self,
        robot_name,
        urdf_filename,
        mesh_path,
        urdf_datadir= './data/urdf',
        batch_size=1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        mesh_nsp=128,
        hand_scale=1.0,
    ):
        self.device = device
        self.robot_name = robot_name
        self.batch_size = batch_size
        # prepare model
        self.robot = pk.build_chain_from_urdf(open(urdf_filename).read()).to(
            dtype=torch.float, device=self.device
        )
        self.robot_full = URDF_PARSER.URDF.from_xml_file(urdf_filename)

        self.global_translation = None
        self.global_rotation = None
        self.softmax = torch.nn.Softmax(dim=-1)

        self.surface_points = {}
        self.surface_points_normal = {}
        visual = URDF.from_xml_string(open(urdf_filename).read())
        self.mesh_verts = {}
        self.mesh_faces = {}

        gripper_surface_pts_dict = os.path.join(
            urdf_datadir, "multidex_gripper_surface_pts.pk"
        )

        with open(gripper_surface_pts_dict, "rb") as f:
            _all_surf_pts_dict = pickle.load(f)
        gripper_surface_pts_info = _all_surf_pts_dict[robot_name]

        self.surface_pts_coords = {}
        # Load the precomputed gripper surface points and normals
        link_keys = gripper_surface_pts_info["points"].keys()

        self.gripper_coords_all = (
            torch.from_numpy(gripper_surface_pts_info["coords_all"]).to(device).float()
        )

        for i_link, link_name in enumerate(link_keys):
            pts = gripper_surface_pts_info["points"][link_name]
            pts_normal = gripper_surface_pts_info["normals"][link_name]
            mesh = gripper_surface_pts_info["meshes"][link_name]
            coords = gripper_surface_pts_info["coords_in_link"][link_name]

            # Make into homog coordinates
            pts = np.concatenate([pts, np.ones([len(pts), 1])], axis=-1)
            pts_normal = np.concatenate(
                [pts_normal, np.ones([len(pts_normal), 1])], axis=-1
            )

            self.surface_points[link_name] = (
                torch.from_numpy(pts)
                .to(device)
                .float()
                .unsqueeze(0)
                .repeat(batch_size, 1, 1)
            )

            self.surface_points_normal[link_name] = (
                torch.from_numpy(pts_normal)
                .to(device)
                .float()
                .unsqueeze(0)
                .repeat(batch_size, 1, 1)
            )

            # No Need to keep a batched version of the coordinates as they are used once
            self.surface_pts_coords[link_name] = (
                torch.from_numpy(coords).to(device).float()
            )

            # visualization mesh
            self.mesh_verts[link_name] = np.array(mesh.vertices)
            self.mesh_faces[link_name] = np.array(mesh.faces)

        # new 2.1
        # Acutally consider both revolute and prismatic joints!
        self.dynamic_joints = []
        for i in range(len(self.robot_full.joints)):
            if self.robot_full.joints[i].joint_type in {"revolute", "prismatic"}:
                self.dynamic_joints.append(self.robot_full.joints[i])
        self.dynamic_joints_q_mid = []
        self.dynamic_joints_q_var = []
        self.dynamic_joints_q_upper = []
        self.dynamic_joints_q_lower = []
        for i in range(len(self.robot.get_joint_parameter_names())):
            for j in range(len(self.dynamic_joints)):
                if (
                    self.dynamic_joints[j].name
                    == self.robot.get_joint_parameter_names()[i]
                ):
                    joint = self.dynamic_joints[j]
            assert joint.name == self.robot.get_joint_parameter_names()[i]
            self.dynamic_joints_q_mid.append(
                (joint.limit.lower + joint.limit.upper) / 2
            )
            self.dynamic_joints_q_var.append(
                ((joint.limit.upper - joint.limit.lower) / 2) ** 2
            )
            self.dynamic_joints_q_lower.append(joint.limit.lower)
            self.dynamic_joints_q_upper.append(joint.limit.upper)

        self.dynamic_joints_q_lower = (
            torch.Tensor(self.dynamic_joints_q_lower)
            .repeat([self.batch_size, 1])
            .to(device)
        )
        self.dynamic_joints_q_upper = (
            torch.Tensor(self.dynamic_joints_q_upper)
            .repeat([self.batch_size, 1])
            .to(device)
        )

        self.current_status = None
        self.scale = hand_scale
        self.palm_normal_dirn = (
            self.get_obj_radius_scale() * self.get_hand_palm_normal()
        )

    def get_obj_radius_scale(self) -> float:
        # scaling factor for object radius
        # obj radius = obj_radius * scale, scale > 1
        gripper = self.robot_name
        if gripper == "Allegro":
            return 1.2
        elif gripper == "Barrett":
            return 1.1
        elif gripper == "fetch_gripper":
            return 1.2
        elif gripper == "franka_panda":
            return 1.2
        elif gripper == "h5_hand":
            return 1.2
        elif gripper == "HumanHand":
            return 1.1
        elif gripper == "jaco_robot":
            return 1.1
        elif gripper == "robotiq_3finger":
            return 1.1
        elif gripper == "sawyer":
            return 1.2
        elif gripper == "shadow_hand":
            return 1.1
        elif gripper == "wsg_50":
            return 1.2
        else:
            return 1

    def get_hand_palm_normal(self):
        gripper = self.robot_name
        if gripper == "Allegro":
            hand_normal = torch.Tensor([[1.0, 0, 0]]).to(self.device).T.float()
        elif gripper == "Barrett":
            hand_normal = torch.Tensor([[0, 0, 1.0]]).to(self.device).T.float()
        elif gripper == "fetch_gripper":
            hand_normal = torch.Tensor([[1.0, 0, 0]]).to(self.device).T.float()
        elif gripper == "franka_panda":
            hand_normal = torch.Tensor([[0, 0, 1.0]]).to(self.device).T.float()
        elif gripper == "h5_hand":
            hand_normal = torch.Tensor([[0, 0, -1.0]]).to(self.device).T.float()
        elif gripper == "HumanHand":
            hand_normal = torch.Tensor([[0, -1.0, 0]]).to(self.device).T.float()
        elif gripper == "jaco_robot":
            hand_normal = torch.Tensor([[-1.0, 0, 0]]).to(self.device).T.float()
        elif gripper == "robotiq_3finger":
            hand_normal = torch.Tensor([[0, 1.0, 0]]).to(self.device).T.float()
        elif gripper == "sawyer":
            hand_normal = torch.Tensor([[0, 0, 1.0]]).to(self.device).T.float()
        elif gripper == "shadow_hand":
            hand_normal = torch.Tensor([[0, -1.0, 0]]).to(self.device).T.float()
        elif gripper == "wsg_50":
            hand_normal = torch.Tensor([[0, 0, 1.0]]).to(self.device).T.float()
        elif gripper == "barrett":  # gdx
            hand_normal = torch.Tensor([[0.0, 0.0, 1.0]]).to(self.device).T.float()
        elif self.robot_name == "allegro_old":
            hand_normal = torch.Tensor([[1.0, 0.0, 0.0]]).to(self.device).T.float()
        elif self.robot_name == "shadowhand":
            hand_normal = torch.Tensor([[0.0, -1.0, 0.0]]).to(self.device).T.float()
        elif self.robot_name == "robotiq_3finger_gdx":
            hand_normal = (
                1.0 * torch.Tensor([[0.0, 0.0, 1.0]]).to(self.device).T.float()
            )
        elif self.robot_name == "ezgripper":
            hand_normal = torch.Tensor([[1.0, 0.0, 0.0]]).to(self.device).T.float()
        elif self.robot_name == "allegro":
            hand_normal = (
                1.0 * torch.Tensor([[1.0, 0.0, 0.0]]).to(self.device).T.float()
            )
        else:
            raise NotImplementedError
        return hand_normal

    def update_kinematics(self, q):
        self.global_translation = q[:, :3]

        self.global_rotation = robust_compute_rotation_matrix_from_ortho6d(q[:, 3:9])
        self.current_status = self.robot.forward_kinematics(q[:, 9:])

    def get_surface_points_prior(self, q=None, downsample=True):
        if q is not None:
            self.update_kinematics(q)
        surface_points = []

        for link_name in self.surface_points:
            # for link_name in parts:
            # get transformation
            trans_matrix = self.current_status[link_name].get_matrix()
            surface_points.append(
                torch.matmul(
                    trans_matrix, self.surface_points[link_name].transpose(1, 2)
                ).transpose(1, 2)[..., :3]
            )
        surface_points = torch.cat(surface_points, 1)
        surface_points = torch.matmul(
            self.global_rotation, surface_points.transpose(1, 2)
        ).transpose(1, 2) + self.global_translation.unsqueeze(1)
        # if downsample:
        #     surface_points = surface_points[:, torch.randperm(surface_points.shape[1])][:, :778]
        return surface_points * self.scale

    def get_surface_points(self, q=None, downsample=True):
        if q is not None:
            self.update_kinematics(q)
        surface_points = []

        for link_name in self.surface_points:
            # for link_name in parts:
            # get transformation
            trans_matrix = self.current_status[link_name].get_matrix()
            surface_points.append(
                torch.matmul(
                    trans_matrix, self.surface_points[link_name].transpose(1, 2)
                ).transpose(1, 2)[..., :3]
            )
        surface_points = torch.cat(surface_points, 1)
        surface_points = torch.matmul(
            self.global_rotation, surface_points.transpose(1, 2)
        ).transpose(1, 2) + self.global_translation.unsqueeze(1)
        return surface_points

    def get_surface_points_new(self, q=None):
        if q is not None:
            self.update_kinematics(q)
        surface_points = []
        for link_name in self.surface_points:
            if self.robot_name == "robotiq_3finger" and link_name == "gripper_palm":
                continue
            if self.robot_name == "robotiq_3finger_real_robot" and link_name == "palm":
                continue
            trans_matrix = self.current_status[link_name].get_matrix()
            surface_points.append(
                torch.matmul(
                    trans_matrix, self.surface_points[link_name].transpose(1, 2)
                ).transpose(1, 2)[..., :3]
            )
        surface_points = torch.cat(surface_points, 1)
        surface_points = torch.matmul(
            self.global_rotation, surface_points.transpose(1, 2)
        ).transpose(1, 2) + self.global_translation.unsqueeze(1)
        return surface_points

    def get_surface_points_palm(self, q=None):
        if q is not None:
            self.update_kinematics(q)
        surface_points = []
        if self.robot_name == "allegro":
            palm_list = ["base_link"]
        elif self.robot_name == "robotiq_3finger_real_robot":
            palm_list = ["palm"]
        else:
            raise NotImplementedError
        for link_name in palm_list:
            # for link_name in parts:
            # get transformation
            trans_matrix = self.current_status[link_name].get_matrix()
            surface_points.append(
                torch.matmul(
                    trans_matrix, self.surface_points[link_name].transpose(1, 2)
                ).transpose(1, 2)[..., :3]
            )
        surface_points = torch.cat(surface_points, 1)
        surface_points = torch.matmul(
            self.global_rotation, surface_points.transpose(1, 2)
        ).transpose(1, 2) + self.global_translation.unsqueeze(1)
        return surface_points

    def get_surface_points_and_normals(self, q=None):
        if q is not None:
            self.update_kinematics(q=q)
        surface_points = []
        surface_normals = []

        for link_name in self.surface_points:
            # for link_name in parts:
            # get transformation
            trans_matrix = self.current_status[link_name].get_matrix()
            surface_points.append(
                torch.matmul(
                    trans_matrix, self.surface_points[link_name].transpose(1, 2)
                ).transpose(1, 2)[..., :3]
            )
            surface_normals.append(
                torch.matmul(
                    trans_matrix, self.surface_points_normal[link_name].transpose(1, 2)
                ).transpose(1, 2)[..., :3]
            )
        surface_points = torch.cat(surface_points, 1)
        surface_normals = torch.cat(surface_normals, 1)
        surface_points = torch.matmul(
            self.global_rotation, surface_points.transpose(1, 2)
        ).transpose(1, 2) + self.global_translation.unsqueeze(1)
        surface_normals = torch.matmul(
            self.global_rotation, surface_normals.transpose(1, 2)
        ).transpose(1, 2)

        return surface_points, surface_normals

    def get_meshes_from_q(self, q=None, i=0):
        data = []
        if q is not None:
            self.update_kinematics(q)
        for idx, link_name in enumerate(self.mesh_verts):
            trans_matrix = self.current_status[link_name].get_matrix()
            trans_matrix = (
                trans_matrix[min(len(trans_matrix) - 1, i)].detach().cpu().numpy()
            )
            v = self.mesh_verts[link_name]
            transformed_v = np.concatenate([v, np.ones([len(v), 1])], axis=-1)
            transformed_v = np.matmul(trans_matrix, transformed_v.T).T[..., :3]
            transformed_v = np.matmul(
                self.global_rotation[i].detach().cpu().numpy(), transformed_v.T
            ).T + np.expand_dims(self.global_translation[i].detach().cpu().numpy(), 0)
            transformed_v = transformed_v * self.scale
            f = self.mesh_faces[link_name]
            data.append(tm.Trimesh(vertices=transformed_v, faces=f))
        return data

    def get_plotly_data(self, q=None, i=0, color="lightblue", opacity=1.0):
        data = []
        if q is not None:
            self.update_kinematics(q)
        for idx, link_name in enumerate(self.mesh_verts):
            trans_matrix = self.current_status[link_name].get_matrix()
            trans_matrix = (
                trans_matrix[min(len(trans_matrix) - 1, i)].detach().cpu().numpy()
            )
            v = self.mesh_verts[link_name]
            transformed_v = np.concatenate([v, np.ones([len(v), 1])], axis=-1)
            transformed_v = np.matmul(trans_matrix, transformed_v.T).T[..., :3]
            transformed_v = np.matmul(
                self.global_rotation[i].detach().cpu().numpy(), transformed_v.T
            ).T + np.expand_dims(self.global_translation[i].detach().cpu().numpy(), 0)
            transformed_v = transformed_v * self.scale
            f = self.mesh_faces[link_name]
            data.append(
                go.Mesh3d(
                    x=transformed_v[:, 0],
                    y=transformed_v[:, 1],
                    z=transformed_v[:, 2],
                    i=f[:, 0],
                    j=f[:, 1],
                    k=f[:, 2],
                    color=color,
                    opacity=opacity,
                )
            )
        return data

    def spherical_distance(self, coords1, coords2, radius=1.0):
        """
        Input
        -----
        - coords1: (N, 2) tensor with (theta, phi) values for N points
        - coords2: (M, 2) tensor with (theta, phi) values for M points
        - radius: float, radius of sphere. default is 1 as we only need relative distances!

        Returns
        -------
        - pairwise_dist: (N, M) tensor with distance equal to the haversine distance
        """
        # Convert coordinates from scaled format to angles in radians
        C1 = coords1.clone()
        C2 = coords2.clone()
        C1[:, 0] *= 2 * torch.pi  # Scale theta
        C2[:, 0] *= 2 * torch.pi
        C1[:, 1] *= torch.pi  # Scale phi
        C2[:, 1] *= torch.pi

        # Calculate differences for haversine formula
        diff = C1[:, None] - C2  # Shape (N, M, 2)
        dtheta = diff[:, :, 0]  # Difference in theta
        dphi = diff[:, :, 1]  # Difference in phi

        phi1 = C1[:, 1]  # phi for the first set of points
        phi2 = C2[:, 1]  # phi for the second set of points

        # Haversine formula
        a = (
            torch.sin(dphi / 2) ** 2
            + torch.cos(phi1.unsqueeze(1))
            * torch.cos(phi2)
            * torch.sin(dtheta / 2) ** 2
        )
        c = 2 * torch.arcsin(torch.sqrt(a))

        # Compute the spherical distance
        distances_haversine = radius * c
        return distances_haversine

    def get_gripper_coords(self):
        return self.gripper_coords_all

    def get_correspondence_mask(self, obj_gcs_pred):
        """
        Establishes a mask over the gripper points for the correspondence-based optimization

        Input:
            obj_gcs_pred: torch tensor, shape (N, 2) array with GCS coordinate contact map prediction for each object point


        Returns:
            mask_corr: integer indices mask over the M gripper surface points for the gripper.
                       shape is (N,)
        """
        with torch.no_grad():
            grp_coord = self.gripper_coords_all
            # valid phi values should be greater than 0.3
            obj_mask = obj_gcs_pred[:, 1] > 0.2
            # obj_pts_idxs = torch.where(obj_mask)
            obj_coord = obj_gcs_pred[obj_mask]
            distances = self.spherical_distance(obj_coord, grp_coord)
            sorted_indices = torch.argsort(distances, dim=1)
            corr_grp_pts_indices = sorted_indices[:, 0]  # pick the closest
            return corr_grp_pts_indices, obj_mask


 

if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from utils.get_models import get_handmodel
    from utils.visualize_plotly import plot_point_cloud
    hand_model = get_handmodel('robotiq_3finger_real_robot', 1, 'cuda', 1.)
    print(len(hand_model.robot.get_joint_parameter_names()))

    joint_lower = np.array(hand_model.revolute_joints_q_lower.cpu().reshape(-1))
    joint_upper = np.array(hand_model.revolute_joints_q_upper.cpu().reshape(-1))
    joint_mid = (joint_lower + joint_upper) / 2
    joints_q = (joint_mid + joint_lower) / 2
    q = torch.from_numpy(np.concatenate([np.array([0, 0, 0, 1, 0, 0, 0, 1, 0]), joints_q])).unsqueeze(0).to(
        device).float()
    # hand_model.get_surface_points(q)
    hand_model.get_surface_points_new(q)
    data = hand_model.get_plotly_data(q=q, opacity=0.5)
    surface_points = hand_model.get_surface_points_new().cpu().squeeze(0)
    data += [plot_point_cloud(surface_points, color='green')]
    fig = go.Figure(data=data)
    fig.show()

    # per dof .gif
    # n_pic = 10
    # outfile_path = 'contents/{}'.format(robot_name)
    # os.makedirs(outfile_path, exist_ok=True)
    # for i_dof in range(len(hand_model.robot.get_joint_parameter_names())):
    #     lower_value = hand_model.revolute_joints_q_lower[i_dof]
    #     upper_value = hand_model.revolute_joints_q_upper[i_dof]
    #     frames = []
    #     for k in range(n_pic):
    #         value = ((n_pic - k) * lower_value + k * upper_value) / n_pic
    #         _q = q.clone()
    #         _q[0, i_dof + 9] = value
    #         data = hand_model.get_plotly_data(_q, opacity=0.5)
    #         im = go.Figure(data=data).to_image()
    #         frames.append(im)
    #         print('{}-{}'.format(i_dof, k))
    #     outfile_gif_name = os.path.join(outfile_path,
    #                                     "{}_{}.gif".format(i_dof, hand_model.robot.get_joint_parameter_names()[i_dof]))
    #     imageio.mimsave(outfile_gif_name, frames, duration=0.1)

