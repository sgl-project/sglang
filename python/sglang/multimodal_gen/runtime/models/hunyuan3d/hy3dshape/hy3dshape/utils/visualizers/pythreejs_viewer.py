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


import numpy as np
from ipywidgets import embed
import pythreejs as p3s
import uuid

from .color_util import get_colors, gen_circle, gen_checkers


EMBED_URL = "https://cdn.jsdelivr.net/npm/@jupyter-widgets/html-manager@1.0.1/dist/embed-amd.js"


class PyThreeJSViewer(object):

    def __init__(self, settings, render_mode="WEBSITE"):
        self.render_mode = render_mode
        self.__update_settings(settings)
        self._light = p3s.DirectionalLight(color='white', position=[0, 0, 1], intensity=0.6)
        self._light2 = p3s.AmbientLight(intensity=0.5)
        self._cam = p3s.PerspectiveCamera(position=[0, 0, 1], lookAt=[0, 0, 0], fov=self.__s["fov"],
                                          aspect=self.__s["width"] / self.__s["height"], children=[self._light])
        self._orbit = p3s.OrbitControls(controlling=self._cam)
        self._scene = p3s.Scene(children=[self._cam, self._light2], background=self.__s["background"])  # "#4c4c80"
        self._renderer = p3s.Renderer(camera=self._cam, scene=self._scene, controls=[self._orbit],
                                      width=self.__s["width"], height=self.__s["height"],
                                      antialias=self.__s["antialias"])

        self.__objects = {}
        self.__cnt = 0

    def jupyter_mode(self):
        self.render_mode = "JUPYTER"

    def offline(self):
        self.render_mode = "OFFLINE"

    def website(self):
        self.render_mode = "WEBSITE"

    def __get_shading(self, shading):
        shad = {"flat": True, "wireframe": False, "wire_width": 0.03, "wire_color": "black",
                "side": 'DoubleSide', "colormap": "viridis", "normalize": [None, None],
                "bbox": False, "roughness": 0.5, "metalness": 0.25, "reflectivity": 1.0,
                "line_width": 1.0, "line_color": "black",
                "point_color": "red", "point_size": 0.01, "point_shape": "circle",
                "text_color": "red"
                }
        for k in shading:
            shad[k] = shading[k]
        return shad

    def __update_settings(self, settings={}):
        sett = {"width": 1600, "height": 800, "antialias": True, "scale": 1.5, "background": "#ffffff",
                "fov": 30}
        for k in settings:
            sett[k] = settings[k]
        self.__s = sett

    def __add_object(self, obj, parent=None):
        if not parent:  # Object is added to global scene and objects dict
            self.__objects[self.__cnt] = obj
            self.__cnt += 1
            self._scene.add(obj["mesh"])
        else:  # Object is added to parent object and NOT to objects dict
            parent.add(obj["mesh"])

        self.__update_view()

        if self.render_mode == "JUPYTER":
            return self.__cnt - 1
        elif self.render_mode == "WEBSITE":
            return self

    def __add_line_geometry(self, lines, shading, obj=None):
        lines = lines.astype("float32", copy=False)
        mi = np.min(lines, axis=0)
        ma = np.max(lines, axis=0)

        geometry = p3s.LineSegmentsGeometry(positions=lines.reshape((-1, 2, 3)))
        material = p3s.LineMaterial(linewidth=shading["line_width"], color=shading["line_color"])
        # , vertexColors='VertexColors'),
        lines = p3s.LineSegments2(geometry=geometry, material=material)  # type='LinePieces')
        line_obj = {"geometry": geometry, "mesh": lines, "material": material,
                    "max": ma, "min": mi, "type": "Lines", "wireframe": None}

        if obj:
            return self.__add_object(line_obj, obj), line_obj
        else:
            return self.__add_object(line_obj)

    def __update_view(self):
        if len(self.__objects) == 0:
            return
        ma = np.zeros((len(self.__objects), 3))
        mi = np.zeros((len(self.__objects), 3))
        for r, obj in enumerate(self.__objects):
            ma[r] = self.__objects[obj]["max"]
            mi[r] = self.__objects[obj]["min"]
        ma = np.max(ma, axis=0)
        mi = np.min(mi, axis=0)
        diag = np.linalg.norm(ma - mi)
        mean = ((ma - mi) / 2 + mi).tolist()
        scale = self.__s["scale"] * (diag)
        self._orbit.target = mean
        self._cam.lookAt(mean)
        self._cam.position = [mean[0], mean[1], mean[2] + scale]
        self._light.position = [mean[0], mean[1], mean[2] + scale]

        self._orbit.exec_three_obj_method('update')
        self._cam.exec_three_obj_method('updateProjectionMatrix')

    def __get_bbox(self, v):
        m = np.min(v, axis=0)
        M = np.max(v, axis=0)

        # Corners of the bounding box
        v_box = np.array([[m[0], m[1], m[2]], [M[0], m[1], m[2]], [M[0], M[1], m[2]], [m[0], M[1], m[2]],
                          [m[0], m[1], M[2]], [M[0], m[1], M[2]], [M[0], M[1], M[2]], [m[0], M[1], M[2]]])

        f_box = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
                          [0, 4], [1, 5], [2, 6], [7, 3]], dtype=np.uint32)
        return v_box, f_box

    def __get_colors(self, v, f, c, sh):
        coloring = "VertexColors"
        if type(c) == np.ndarray and c.size == 3:  # Single color
            colors = np.ones_like(v)
            colors[:, 0] = c[0]
            colors[:, 1] = c[1]
            colors[:, 2] = c[2]
            # print("Single colors")
        elif type(c) == np.ndarray and len(c.shape) == 2 and c.shape[1] == 3:  # Color values for
            if c.shape[0] == f.shape[0]:  # faces
                colors = np.hstack([c, c, c]).reshape((-1, 3))
                coloring = "FaceColors"
                # print("Face color values")
            elif c.shape[0] == v.shape[0]:  # vertices
                colors = c
                # print("Vertex color values")
            else:  # Wrong size, fallback
                print("Invalid color array given! Supported are numpy arrays.", type(c))
                colors = np.ones_like(v)
                colors[:, 0] = 1.0
                colors[:, 1] = 0.874
                colors[:, 2] = 0.0
        elif type(c) == np.ndarray and c.size == f.shape[0]:  # Function values for faces
            normalize = sh["normalize"][0] != None and sh["normalize"][1] != None
            cc = get_colors(c, sh["colormap"], normalize=normalize,
                            vmin=sh["normalize"][0], vmax=sh["normalize"][1])
            # print(cc.shape)
            colors = np.hstack([cc, cc, cc]).reshape((-1, 3))
            coloring = "FaceColors"
            # print("Face function values")
        elif type(c) == np.ndarray and c.size == v.shape[0]:  # Function values for vertices
            normalize = sh["normalize"][0] != None and sh["normalize"][1] != None
            colors = get_colors(c, sh["colormap"], normalize=normalize,
                                vmin=sh["normalize"][0], vmax=sh["normalize"][1])
            # print("Vertex function values")

        else:
            colors = np.ones_like(v)
            colors[:, 0] = 1.0
            colors[:, 1] = 0.874
            colors[:, 2] = 0.0

            # No color
            if c is not None:
                print("Invalid color array given! Supported are numpy arrays.", type(c))

        return colors, coloring

    def __get_point_colors(self, v, c, sh):
        v_color = True
        if c is None:  # No color given, use global color
            # conv = mpl.colors.ColorConverter()
            colors = sh["point_color"]  # np.array(conv.to_rgb(sh["point_color"]))
            v_color = False
        elif isinstance(c, str):  # No color given, use global color
            # conv = mpl.colors.ColorConverter()
            colors = c  # np.array(conv.to_rgb(c))
            v_color = False
        elif type(c) == np.ndarray and len(c.shape) == 2 and c.shape[0] == v.shape[0] and c.shape[1] == 3:
            # Point color
            colors = c.astype("float32", copy=False)

        elif isinstance(c, np.ndarray) and len(c.shape) == 2 and c.shape[0] == v.shape[0] and c.shape[1] != 3:
            # Function values for vertices, but the colors are features
            c_norm = np.linalg.norm(c, ord=2, axis=-1)
            normalize = sh["normalize"][0] != None and sh["normalize"][1] != None
            colors = get_colors(c_norm, sh["colormap"], normalize=normalize,
                                vmin=sh["normalize"][0], vmax=sh["normalize"][1])
            colors = colors.astype("float32", copy=False)

        elif type(c) == np.ndarray and c.size == v.shape[0]:  # Function color
            normalize = sh["normalize"][0] != None and sh["normalize"][1] != None
            colors = get_colors(c, sh["colormap"], normalize=normalize,
                                vmin=sh["normalize"][0], vmax=sh["normalize"][1])
            colors = colors.astype("float32", copy=False)
            # print("Vertex function values")

        else:
            print("Invalid color array given! Supported are numpy arrays.", type(c))
            colors = sh["point_color"]
            v_color = False

        return colors, v_color

    def add_mesh(self, v, f, c=None, uv=None, n=None, shading={}, texture_data=None, **kwargs):
        shading.update(kwargs)
        sh = self.__get_shading(shading)
        mesh_obj = {}

        # it is a tet
        if v.shape[1] == 3 and f.shape[1] == 4:
            f_tmp = np.ndarray([f.shape[0] * 4, 3], dtype=f.dtype)
            for i in range(f.shape[0]):
                f_tmp[i * 4 + 0] = np.array([f[i][1], f[i][0], f[i][2]])
                f_tmp[i * 4 + 1] = np.array([f[i][0], f[i][1], f[i][3]])
                f_tmp[i * 4 + 2] = np.array([f[i][1], f[i][2], f[i][3]])
                f_tmp[i * 4 + 3] = np.array([f[i][2], f[i][0], f[i][3]])
            f = f_tmp

        if v.shape[1] == 2:
            v = np.append(v, np.zeros([v.shape[0], 1]), 1)

        # Type adjustment vertices
        v = v.astype("float32", copy=False)

        # Color setup
        colors, coloring = self.__get_colors(v, f, c, sh)

        # Type adjustment faces and colors
        c = colors.astype("float32", copy=False)

        # Material and geometry setup
        ba_dict = {"color": p3s.BufferAttribute(c)}
        if coloring == "FaceColors":
            verts = np.zeros((f.shape[0] * 3, 3), dtype="float32")
            for ii in range(f.shape[0]):
                # print(ii*3, f[ii])
                verts[ii * 3] = v[f[ii, 0]]
                verts[ii * 3 + 1] = v[f[ii, 1]]
                verts[ii * 3 + 2] = v[f[ii, 2]]
            v = verts
        else:
            f = f.astype("uint32", copy=False).ravel()
            ba_dict["index"] = p3s.BufferAttribute(f, normalized=False)

        ba_dict["position"] = p3s.BufferAttribute(v, normalized=False)

        if uv is not None:
            uv = (uv - np.min(uv)) / (np.max(uv) - np.min(uv))
            if texture_data is None:
                texture_data = gen_checkers(20, 20)
            tex = p3s.DataTexture(data=texture_data, format="RGBFormat", type="FloatType")
            material = p3s.MeshStandardMaterial(map=tex, reflectivity=sh["reflectivity"], side=sh["side"],
                                                roughness=sh["roughness"], metalness=sh["metalness"],
                                                flatShading=sh["flat"],
                                                polygonOffset=True, polygonOffsetFactor=1, polygonOffsetUnits=5)
            ba_dict["uv"] = p3s.BufferAttribute(uv.astype("float32", copy=False))
        else:
            material = p3s.MeshStandardMaterial(vertexColors=coloring, reflectivity=sh["reflectivity"],
                                                side=sh["side"], roughness=sh["roughness"], metalness=sh["metalness"],
                                                flatShading=sh["flat"],
                                                polygonOffset=True, polygonOffsetFactor=1, polygonOffsetUnits=5)

        if type(n) != type(None) and coloring == "VertexColors":  # TODO: properly handle normals for FaceColors as well
            ba_dict["normal"] = p3s.BufferAttribute(n.astype("float32", copy=False), normalized=True)

        geometry = p3s.BufferGeometry(attributes=ba_dict)

        if coloring == "VertexColors" and type(n) == type(None):
            geometry.exec_three_obj_method('computeVertexNormals')
        elif coloring == "FaceColors" and type(n) == type(None):
            geometry.exec_three_obj_method('computeFaceNormals')

        # Mesh setup
        mesh = p3s.Mesh(geometry=geometry, material=material)

        # Wireframe setup
        mesh_obj["wireframe"] = None
        if sh["wireframe"]:
            wf_geometry = p3s.WireframeGeometry(mesh.geometry)  # WireframeGeometry
            wf_material = p3s.LineBasicMaterial(color=sh["wire_color"], linewidth=sh["wire_width"])
            wireframe = p3s.LineSegments(wf_geometry, wf_material)
            mesh.add(wireframe)
            mesh_obj["wireframe"] = wireframe

        # Bounding box setup
        if sh["bbox"]:
            v_box, f_box = self.__get_bbox(v)
            _, bbox = self.add_edges(v_box, f_box, sh, mesh)
            mesh_obj["bbox"] = [bbox, v_box, f_box]

        # Object setup
        mesh_obj["max"] = np.max(v, axis=0)
        mesh_obj["min"] = np.min(v, axis=0)
        mesh_obj["geometry"] = geometry
        mesh_obj["mesh"] = mesh
        mesh_obj["material"] = material
        mesh_obj["type"] = "Mesh"
        mesh_obj["shading"] = sh
        mesh_obj["coloring"] = coloring
        mesh_obj["arrays"] = [v, f, c]  # TODO replays with proper storage or remove if not needed

        return self.__add_object(mesh_obj)

    def add_lines(self, beginning, ending, shading={}, obj=None, **kwargs):
        shading.update(kwargs)
        if len(beginning.shape) == 1:
            if len(beginning) == 2:
                beginning = np.array([[beginning[0], beginning[1], 0]])
        else:
            if beginning.shape[1] == 2:
                beginning = np.append(
                    beginning, np.zeros([beginning.shape[0], 1]), 1)
        if len(ending.shape) == 1:
            if len(ending) == 2:
                ending = np.array([[ending[0], ending[1], 0]])
        else:
            if ending.shape[1] == 2:
                ending = np.append(
                    ending, np.zeros([ending.shape[0], 1]), 1)

        sh = self.__get_shading(shading)
        lines = np.hstack([beginning, ending])
        lines = lines.reshape((-1, 3))
        return self.__add_line_geometry(lines, sh, obj)

    def add_edges(self, vertices, edges, shading={}, obj=None, **kwargs):
        shading.update(kwargs)
        if vertices.shape[1] == 2:
            vertices = np.append(
                vertices, np.zeros([vertices.shape[0], 1]), 1)
        sh = self.__get_shading(shading)
        lines = np.zeros((edges.size, 3))
        cnt = 0
        for e in edges:
            lines[cnt, :] = vertices[e[0]]
            lines[cnt + 1, :] = vertices[e[1]]
            cnt += 2
        return self.__add_line_geometry(lines, sh, obj)

    def add_points(self, points, c=None, shading={}, obj=None, **kwargs):
        shading.update(kwargs)
        if len(points.shape) == 1:
            if len(points) == 2:
                points = np.array([[points[0], points[1], 0]])
        else:
            if points.shape[1] == 2:
                points = np.append(
                    points, np.zeros([points.shape[0], 1]), 1)
        sh = self.__get_shading(shading)
        points = points.astype("float32", copy=False)
        mi = np.min(points, axis=0)
        ma = np.max(points, axis=0)

        g_attributes = {"position": p3s.BufferAttribute(points, normalized=False)}
        m_attributes = {"size": sh["point_size"]}

        if sh["point_shape"] == "circle":  # Plot circles
            tex = p3s.DataTexture(data=gen_circle(16, 16), format="RGBAFormat", type="FloatType")
            m_attributes["map"] = tex
            m_attributes["alphaTest"] = 0.5
            m_attributes["transparency"] = True
        else:  # Plot squares
            pass

        colors, v_colors = self.__get_point_colors(points, c, sh)
        if v_colors:  # Colors per point
            m_attributes["vertexColors"] = 'VertexColors'
            g_attributes["color"] = p3s.BufferAttribute(colors, normalized=False)

        else:  # Colors for all points
            m_attributes["color"] = colors

        material = p3s.PointsMaterial(**m_attributes)
        geometry = p3s.BufferGeometry(attributes=g_attributes)
        points = p3s.Points(geometry=geometry, material=material)
        point_obj = {"geometry": geometry, "mesh": points, "material": material,
                     "max": ma, "min": mi, "type": "Points", "wireframe": None}

        if obj:
            return self.__add_object(point_obj, obj), point_obj
        else:
            return self.__add_object(point_obj)

    def remove_object(self, obj_id):
        if obj_id not in self.__objects:
            print("Invalid object id. Valid ids are: ", list(self.__objects.keys()))
            return
        self._scene.remove(self.__objects[obj_id]["mesh"])
        del self.__objects[obj_id]
        self.__update_view()

    def reset(self):
        for obj_id in list(self.__objects.keys()).copy():
            self._scene.remove(self.__objects[obj_id]["mesh"])
            del self.__objects[obj_id]
        self.__update_view()

    def update_object(self, oid=0, vertices=None, colors=None, faces=None):
        obj = self.__objects[oid]
        if type(vertices) != type(None):
            if obj["coloring"] == "FaceColors":
                f = obj["arrays"][1]
                verts = np.zeros((f.shape[0] * 3, 3), dtype="float32")
                for ii in range(f.shape[0]):
                    # print(ii*3, f[ii])
                    verts[ii * 3] = vertices[f[ii, 0]]
                    verts[ii * 3 + 1] = vertices[f[ii, 1]]
                    verts[ii * 3 + 2] = vertices[f[ii, 2]]
                v = verts

            else:
                v = vertices.astype("float32", copy=False)
            obj["geometry"].attributes["position"].array = v
            # self.wireframe.attributes["position"].array = v # Wireframe updates?
            obj["geometry"].attributes["position"].needsUpdate = True
        #           obj["geometry"].exec_three_obj_method('computeVertexNormals')
        if type(colors) != type(None):
            colors, coloring = self.__get_colors(obj["arrays"][0], obj["arrays"][1], colors, obj["shading"])
            colors = colors.astype("float32", copy=False)
            obj["geometry"].attributes["color"].array = colors
            obj["geometry"].attributes["color"].needsUpdate = True
        if type(faces) != type(None):
            if obj["coloring"] == "FaceColors":
                print("Face updates are currently only possible in vertex color mode.")
                return
            f = faces.astype("uint32", copy=False).ravel()
            print(obj["geometry"].attributes)
            obj["geometry"].attributes["index"].array = f
            # self.wireframe.attributes["position"].array = v # Wireframe updates?
            obj["geometry"].attributes["index"].needsUpdate = True
        #            obj["geometry"].exec_three_obj_method('computeVertexNormals')
        # self.mesh.geometry.verticesNeedUpdate = True
        # self.mesh.geometry.elementsNeedUpdate = True
        # self.update()
        if self.render_mode == "WEBSITE":
            return self

    #    def update(self):
    #        self.mesh.exec_three_obj_method('update')
    #        self.orbit.exec_three_obj_method('update')
    #        self.cam.exec_three_obj_method('updateProjectionMatrix')
    #        self.scene.exec_three_obj_method('update')

    def add_text(self, text, shading={}, **kwargs):
        shading.update(kwargs)
        sh = self.__get_shading(shading)
        tt = p3s.TextTexture(string=text, color=sh["text_color"])
        sm = p3s.SpriteMaterial(map=tt)
        text = p3s.Sprite(material=sm, scaleToTexture=True)
        self._scene.add(text)

    # def add_widget(self, widget, callback):
    #    self.widgets.append(widget)
    #    widget.observe(callback, names='value')

    #    def add_dropdown(self, options, default, desc, cb):
    #        widget = widgets.Dropdown(options=options, value=default, description=desc)
    #        self.__widgets.append(widget)
    #        widget.observe(cb, names="value")
    #        display(widget)

    #    def add_button(self, text, cb):
    #        button = widgets.Button(description=text)
    #        self.__widgets.append(button)
    #        button.on_click(cb)
    #        display(button)

    def to_html(self, imports=True, html_frame=True):
        # Bake positions (fixes centering bug in offline rendering)
        if len(self.__objects) == 0:
            return
        ma = np.zeros((len(self.__objects), 3))
        mi = np.zeros((len(self.__objects), 3))
        for r, obj in enumerate(self.__objects):
            ma[r] = self.__objects[obj]["max"]
            mi[r] = self.__objects[obj]["min"]
        ma = np.max(ma, axis=0)
        mi = np.min(mi, axis=0)
        diag = np.linalg.norm(ma - mi)
        mean = (ma - mi) / 2 + mi
        for r, obj in enumerate(self.__objects):
            v = self.__objects[obj]["geometry"].attributes["position"].array
            v -= mean
            # v += np.array([0.0, .9, 0.0]) #! to move the obj to the center of window

        scale = self.__s["scale"] * (diag)
        self._orbit.target = [0.0, 0.0, 0.0]
        self._cam.lookAt([0.0, 0.0, 0.0])
        # self._cam.position = [0.0, 0.0, scale]
        self._cam.position = [0.0, 0.5, scale * 1.3] #! show four complete meshes in the window
        self._light.position = [0.0, 0.0, scale]

        state = embed.dependency_state(self._renderer)

        # Somehow these entries are missing when the state is exported in python.
        # Exporting from the GUI works, so we are inserting the missing entries.
        for k in state:
            if state[k]["model_name"] == "OrbitControlsModel":
                state[k]["state"]["maxAzimuthAngle"] = "inf"
                state[k]["state"]["maxDistance"] = "inf"
                state[k]["state"]["maxZoom"] = "inf"
                state[k]["state"]["minAzimuthAngle"] = "-inf"

        tpl = embed.load_requirejs_template
        if not imports:
            embed.load_requirejs_template = ""

        s = embed.embed_snippet(self._renderer, state=state, embed_url=EMBED_URL)
        # s = embed.embed_snippet(self.__w, state=state)
        embed.load_requirejs_template = tpl

        if html_frame:
            s = "<html>\n<body>\n" + s + "\n</body>\n</html>"

        # Revert changes
        for r, obj in enumerate(self.__objects):
            v = self.__objects[obj]["geometry"].attributes["position"].array
            v += mean
        self.__update_view()

        return s

    def save(self, filename=""):
        if filename == "":
            uid = str(uuid.uuid4()) + ".html"
        else:
            filename = filename.replace(".html", "")
            uid = filename + '.html'
        with open(uid, "w") as f:
            f.write(self.to_html())
        print("Plot saved to file %s." % uid)
