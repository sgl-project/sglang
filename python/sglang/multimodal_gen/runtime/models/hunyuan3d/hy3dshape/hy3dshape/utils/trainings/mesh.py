# -*- coding: utf-8 -*-

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import os
import cv2
import numpy as np
import PIL.Image
from typing import Optional

import trimesh


def save_obj(pointnp_px3, facenp_fx3, fname):
    fid = open(fname, "w")
    write_str = ""
    for pidx, p in enumerate(pointnp_px3):
        pp = p
        write_str += "v %f %f %f\n" % (pp[0], pp[1], pp[2])

    for i, f in enumerate(facenp_fx3):
        f1 = f + 1
        write_str += "f %d %d %d\n" % (f1[0], f1[1], f1[2])
    fid.write(write_str)
    fid.close()
    return


def savemeshtes2(pointnp_px3, tcoords_px2, facenp_fx3, facetex_fx3, tex_map, fname):
    fol, na = os.path.split(fname)
    na, _ = os.path.splitext(na)

    matname = "%s/%s.mtl" % (fol, na)
    fid = open(matname, "w")
    fid.write("newmtl material_0\n")
    fid.write("Kd 1 1 1\n")
    fid.write("Ka 0 0 0\n")
    fid.write("Ks 0.4 0.4 0.4\n")
    fid.write("Ns 10\n")
    fid.write("illum 2\n")
    fid.write("map_Kd %s.png\n" % na)
    fid.close()
    ####

    fid = open(fname, "w")
    fid.write("mtllib %s.mtl\n" % na)

    for pidx, p3 in enumerate(pointnp_px3):
        pp = p3
        fid.write("v %f %f %f\n" % (pp[0], pp[1], pp[2]))

    for pidx, p2 in enumerate(tcoords_px2):
        pp = p2
        fid.write("vt %f %f\n" % (pp[0], pp[1]))

    fid.write("usemtl material_0\n")
    for i, f in enumerate(facenp_fx3):
        f1 = f + 1
        f2 = facetex_fx3[i] + 1
        fid.write("f %d/%d %d/%d %d/%d\n" % (f1[0], f2[0], f1[1], f2[1], f1[2], f2[2]))
    fid.close()

    PIL.Image.fromarray(np.ascontiguousarray(tex_map), "RGB").save(
        os.path.join(fol, "%s.png" % na))

    return


class MeshOutput(object):

    def __init__(self,
                 mesh_v: np.ndarray,
                 mesh_f: np.ndarray,
                 vertex_colors: Optional[np.ndarray] = None,
                 uvs: Optional[np.ndarray] = None,
                 mesh_tex_idx: Optional[np.ndarray] = None,
                 tex_map: Optional[np.ndarray] = None):

        self.mesh_v = mesh_v
        self.mesh_f = mesh_f
        self.vertex_colors = vertex_colors
        self.uvs = uvs
        self.mesh_tex_idx = mesh_tex_idx
        self.tex_map = tex_map

    def contain_uv_texture(self):
        return (self.uvs is not None) and (self.mesh_tex_idx is not None) and (self.tex_map is not None)

    def contain_vertex_colors(self):
        return self.vertex_colors is not None

    def export(self, fname):

        if self.contain_uv_texture():
            savemeshtes2(
                self.mesh_v,
                self.uvs,
                self.mesh_f,
                self.mesh_tex_idx,
                self.tex_map,
                fname
            )

        elif self.contain_vertex_colors():
            mesh_obj = trimesh.Trimesh(vertices=self.mesh_v, faces=self.mesh_f, vertex_colors=self.vertex_colors)
            mesh_obj.export(fname)

        else:
            save_obj(
                self.mesh_v,
                self.mesh_f,
                fname
            )



