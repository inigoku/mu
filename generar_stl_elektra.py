from __future__ import annotations

import argparse
import os
import ssl
import tempfile
import time
from pathlib import Path

# Evitar RecursionError de pip_system_certs + truststore + httpx2 en Python 3.14
try:
    import truststore
    truststore.extract_from_ssl()
    truststore.SSLContext = ssl.SSLContext  # type: ignore[misc]
except Exception:
    pass

import numpy as np
from PIL import Image, ImageDraw
import requests
from scipy import ndimage
import trimesh


DEFAULT_INPUT = Path(
    r"C:\git\test\MarvelUnitedPromos\images\Temporada_1\PROMOS_STRETCH_GOALS\Elektra\figurita.jpg"
)
ENV_FILE = Path(r"C:\git\test\.env")


def load_env_key(var_name: str = "TRIPO_API_KEY") -> str | None:
    if os.environ.get(var_name):
        return os.environ[var_name]
    if ENV_FILE.is_file():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(f"{var_name}="):
                return line.split("=", 1)[1].strip()
    return None


def extract_clean_views(
    image_path: Path,
    out_front: Path,
    out_back: Path,
    split_x: int = 317,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Separa la imagen compuesta (frontal + trasera) y elimina el fondo
    de puntos/líneas grises conservando los huecos internos (como entre las piernas).
    Devuelve arrays RGBA (H, W, 4) centrados respecto al centro de la peana.
    """
    im = Image.open(image_path).convert("RGB")
    arr = np.array(im, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # El render de la miniatura tiene dominante azul-cian (b - r alto);
    # el fondo blanco y la trama de puntos grises tienen b - r <= 14.
    chroma = b - r
    raw_mask = chroma > 20.0
    # Eliminar puntos de la trama que rozan el contorno
    raw_mask = ndimage.binary_opening(raw_mask, structure=np.ones((2, 2)))

    # Conservar únicamente las 2 componentes conexas principales (frontal y trasera)
    labeled, num = ndimage.label(raw_mask)
    sizes = ndimage.sum(raw_mask, labeled, range(1, num + 1))
    top2 = np.argsort(sizes)[-2:] + 1
    fg_mask = np.isin(labeled, top2)

    # Rellenar únicamente micro-poros internos (< 35 píxeles) para NO tapar
    # el hueco entre las piernas ni entre los mechones de pelo
    inv_mask = ~fg_mask
    holes_lbl, num_holes = ndimage.label(inv_mask)
    if num_holes > 0:
        hole_sizes = ndimage.sum(inv_mask, holes_lbl, range(1, num_holes + 1))
        small_holes = np.where(hole_sizes < 35)[0] + 1
        fg_mask = fg_mask | np.isin(holes_lbl, small_holes)

    alpha = np.where(fg_mask, 1.0, 0.0).astype(np.float32)
    alpha_u8 = (ndimage.gaussian_filter(alpha, sigma=0.5) * 255).astype(np.uint8)
    alpha_u8 = np.where(fg_mask, np.maximum(alpha_u8, 225), 0).astype(np.uint8)

    rgba = np.dstack([arr.astype(np.uint8), alpha_u8])
    front_crop = rgba[:, :split_x]
    back_crop = rgba[:, split_x + 1 :]

    # Centrar ambas vistas en un lienzo común (360 x 330) alineado por el centro de la peana
    canvas_h, canvas_w = 330, 360
    target_cx = canvas_w // 2

    def center_on_base(crop: np.ndarray) -> np.ndarray:
        mask_c = crop[:, :, 3] > 127
        base_xs = np.where(mask_c[292:308, :])[1]
        cx = int(round(0.5 * (base_xs.min() + base_xs.max())))
        out = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
        shift_x = target_cx - cx
        shift_y = 4

        src_x0 = max(0, -shift_x)
        dst_x0 = max(0, shift_x)
        copy_w = min(crop.shape[1] - src_x0, canvas_w - dst_x0)

        out[shift_y : shift_y + crop.shape[0], dst_x0 : dst_x0 + copy_w] = crop[
            :, src_x0 : src_x0 + copy_w
        ]
        return out

    front_centered = center_on_base(front_crop)
    back_centered = center_on_base(back_crop)

    Image.fromarray(front_centered).save(out_front)
    Image.fromarray(back_centered).save(out_back)
    print(f"[OK] Vistas limpias guardadas en:\n  - {out_front}\n  - {out_back}")
    return front_centered, back_centered


def try_huggingface_multiview(
    front_png: Path,
    back_png: Path,
    output_stl: Path,
    hf_token: str | None = None,
) -> bool:
    """
    Intenta usar el Space oficial Multi-Vista tencent/Hunyuan3D-2mv en Hugging Face
    y convierte la malla resultante (.glb/.obj) a .stl.
    """
    from gradio_client import Client, handle_file

    spaces = ["tencent/Hunyuan3D-2mv", "tencent/Hunyuan3D-2"]
    for space in spaces:
        try:
            print(f"[Cloud] Conectando con Hugging Face Space: {space} ...")
            client = Client(space, token=hf_token)
            result = client.predict(
                image=handle_file(str(front_png)),
                mv_image_front=handle_file(str(front_png)),
                mv_image_back=handle_file(str(back_png)),
                mv_image_left=None,
                mv_image_right=None,
                steps=30,
                guidance_scale=5.0,
                seed=1234,
                octree_resolution=256,
                check_box_rembg=False,
                num_chunks=8000,
                randomize_seed=False,
                api_name="/shape_generation",
            )
            mesh_path = result[0] if isinstance(result, (list, tuple)) else result
            if isinstance(mesh_path, dict) and "value" in mesh_path:
                mesh_path = mesh_path["value"]

            loaded = trimesh.load(mesh_path, force="mesh")
            loaded.export(output_stl)
            print(f"[OK] Modelo IA Multi-Vista exportado a: {output_stl}")
            return True
        except Exception as exc:
            print(f"[Aviso] No se pudo conectar a {space} ({type(exc).__name__}: {exc})")
    return False


def convert_mesh_to_stl(
    input_mesh_path: Path,
    output_stl: Path,
    base_size_mm: float = 30.0,
) -> trimesh.Trimesh:
    """
    Convierte un archivo 3D (.glb, .obj, .3mf) descargado de Tripo3D Studio o MakerWorld
    a un archivo .STL orientado y escalado con peana de 30x30 mm listo para laminar.
    """
    mesh = trimesh.load(input_mesh_path, force="mesh")
    # En GLB (glTF), el eje vertical suele ser Y; pasamos a Z vertical para slicers
    v = mesh.vertices.copy()
    ext = mesh.extents
    if ext[1] > ext[2]:
        mesh.vertices = np.column_stack([v[:, 0], -v[:, 2], v[:, 1]])

    bounds = mesh.bounds
    mesh.vertices[:, 2] -= bounds[0, 2]
    total_h = mesh.extents[2]

    # Medir los vértices de la peana (10% inferior de la altura) y escalar a 30x30 mm
    base_mask = mesh.vertices[:, 2] <= (0.10 * total_h)
    base_verts = mesh.vertices[base_mask]
    base_w = float(base_verts[:, 0].max() - base_verts[:, 0].min())
    base_d = float(base_verts[:, 1].max() - base_verts[:, 1].min())

    sx = base_size_mm / max(base_w, 1e-5)
    sy = base_size_mm / max(base_d, 1e-5)
    sz = sx  # Escala vertical proporcional al diámetro de peana

    base_cx = 0.5 * float(base_verts[:, 0].max() + base_verts[:, 0].min())
    base_cy = 0.5 * float(base_verts[:, 1].max() + base_verts[:, 1].min())
    mesh.vertices[:, 0] = (mesh.vertices[:, 0] - base_cx) * sx
    mesh.vertices[:, 1] = (mesh.vertices[:, 1] - base_cy) * sy
    mesh.vertices[:, 2] *= sz

    mesh.export(output_stl)
    print(f"[OK] Malla convertida con peana {base_size_mm:.1f}x{base_size_mm:.1f} mm en: {output_stl}")
    return mesh


def fuse_front_back_meshes(
    front_mesh_path: Path,
    back_mesh_path: Path,
    output_stl: Path,
    base_size_mm: float = 30.0,
) -> trimesh.Trimesh:
    """
    Si generas 2 modelos gratuitos de 1 sola imagen en Tripo3D (uno con elektra_frontal.png
    y otro con elektra_trasera.png), esta función alinea ambos sobre la peana de 30x30 mm,
    rota 180° el modelo trasero y los fusiona en un único .STL.
    """
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tf, tempfile.NamedTemporaryFile(
        suffix=".stl", delete=False
    ) as tb:
        f_tmp, b_tmp = Path(tf.name), Path(tb.name)

    m_front = convert_mesh_to_stl(front_mesh_path, f_tmp, base_size_mm)
    m_back = convert_mesh_to_stl(back_mesh_path, b_tmp, base_size_mm)

    # Rotar 180 grados en Z el modelo generado desde la vista trasera
    m_back.vertices[:, 0] = -m_back.vertices[:, 0]
    m_back.vertices[:, 1] = -m_back.vertices[:, 1]

    # Conservar la mitad delantera (Y <= 1.5 mm) de m_front y la mitad trasera (Y >= -1.5 mm) de m_back
    c_front = m_front.triangles_center
    c_back = m_back.triangles_center

    faces_f = m_front.faces[c_front[:, 1] <= 1.5]
    faces_b = m_back.faces[c_back[:, 1] >= -1.5]

    sub_f = trimesh.Trimesh(vertices=m_front.vertices, faces=faces_f, process=True)
    sub_b = trimesh.Trimesh(vertices=m_back.vertices, faces=faces_b, process=True)

    fused = trimesh.util.concatenate([sub_f, sub_b])
    fused.export(output_stl)
    print(
        f"[OK] Mallas frontal y trasera fusionadas con peana {base_size_mm:.1f}x{base_size_mm:.1f} mm en: {output_stl}"
    )
    return fused


def try_tripo3d_multiview(
    front_png: Path,
    back_png: Path,
    output_stl: Path,
    api_key: str,
    target_height_mm: float = 42.0,
) -> trimesh.Trimesh | None:
    """
    Usa la API oficial de Tripo3D (https://api.tripo3d.ai/v2/openapi, compatible con Zscaler)
    para generar el modelo 3D a partir de las vistas frontal y trasera.
    """
    headers = {"Authorization": f"Bearer {api_key}"}
    base_url = "https://api.tripo3d.ai/v2/openapi"

    print("[Tripo3D] Subiendo vistas frontal y trasera a api.tripo3d.ai ...")
    with open(front_png, "rb") as f1:
        r1 = requests.post(
            f"{base_url}/upload",
            headers=headers,
            files={"file": (front_png.name, f1, "image/png")},
            timeout=30,
        )
    with open(back_png, "rb") as f2:
        r2 = requests.post(
            f"{base_url}/upload",
            headers=headers,
            files={"file": (back_png.name, f2, "image/png")},
            timeout=30,
        )

    if r1.status_code != 200 or r2.status_code != 200:
        print(f"[Aviso Tripo3D] Error subiendo imágenes: {r1.text} / {r2.text}")
        return None

    t_front = r1.json()["data"]["image_token"]
    t_back = r2.json()["data"]["image_token"]

    payload = {
        "type": "multiview_to_model",
        "files": [
            {"type": "png", "file_token": t_front},
            {},
            {"type": "png", "file_token": t_back},
            {},
        ],
    }
    r_task = requests.post(
        f"{base_url}/task", headers=headers, json=payload, timeout=30
    )
    if r_task.status_code != 200 or r_task.json().get("code") != 0:
        print(f"[Aviso Tripo3D] No se pudo crear la tarea API: {r_task.text}")
        return None

    task_id = r_task.json()["data"]["task_id"]
    print(f"[Tripo3D] Tarea creada ({task_id}). Esperando generación 3D...")

    for _ in range(90):
        time.sleep(3)
        r_poll = requests.get(
            f"{base_url}/task/{task_id}", headers=headers, timeout=20
        )
        data = r_poll.json().get("data", {})
        status = data.get("status")
        progress = data.get("progress", 0)
        print(f"  - Estado: {status} ({progress}%)")
        if status == "success":
            Output = data.get("output", {})
            model_url = (
                Output.get("model")
                or Output.get("pbr_model")
                or Output.get("base_model")
            )
            if not model_url:
                print(f"[Aviso Tripo3D] Respuesta sin URL de modelo: {data}")
                return None
            r_dl = requests.get(model_url, timeout=60)
            with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
                tmp.write(r_dl.content)
                tmp_path = Path(tmp.name)
            return convert_mesh_to_stl(tmp_path, output_stl, target_height_mm)
        if status in ("failed", "cancelled", "expired"):
            print(f"[Aviso Tripo3D] La tarea terminó con estado: {status}")
            return None

    print("[Aviso Tripo3D] Tiempo de espera agotado.")
    return None


def build_bilateral_3d_stl(
    front_rgba: np.ndarray,
    back_rgba: np.ndarray,
    output_stl: Path,
    base_size_mm: float = 30.0,
) -> trimesh.Trimesh:
    """
    Reconstruye una miniatura 3D 100% cerrada (watertight) combinando:
      1. Silueta frontal y trasera (espejada respecto al eje central de la peana).
      2. Volumen anatómico suave por transformada de distancia multi-escala
         + envolvente cilíndrica en la peana inferior.
      3. Shape-from-Shading bilateral sin deformación horizontal (conservando
         íntegramente el esculpido del pelo, nudo de cinta, espalda y túnica trasera).
    """
    print("[Local 3D] Construyendo malla 3D bilateral de alta resolución...")

    scale = 2
    H0, W0, _ = front_rgba.shape
    H, W = H0 * scale, W0 * scale

    front_up = np.array(
        Image.fromarray(front_rgba).resize((W, H), Image.Resampling.BICUBIC),
        dtype=np.float32,
    )
    back_mirror_pil = Image.fromarray(back_rgba[:, ::-1, :]).resize(
        (W, H), Image.Resampling.BICUBIC
    )
    back_up = np.array(back_mirror_pil, dtype=np.float32)

    f_mask = front_up[:, :, 3] > 140
    b_mask = back_up[:, :, 3] > 140

    # Limpiar la máscara frontal para que sea un único sólido conexo suave
    f_mask = ndimage.binary_opening(f_mask, structure=np.ones((3, 3)))
    lbl, num = ndimage.label(f_mask)
    if num > 1:
        sizes = ndimage.sum(f_mask, lbl, range(1, num + 1))
        f_mask = lbl == (np.argmax(sizes) + 1)

    def get_luminance(rgba: np.ndarray, mask: np.ndarray) -> np.ndarray:
        lum = (
            0.2126 * rgba[:, :, 0]
            + 0.7152 * rgba[:, :, 1]
            + 0.0722 * rgba[:, :, 2]
        ) / 255.0
        vals = lum[mask]
        p_low, p_high = np.percentile(vals, [2, 98])
        lum = np.clip((lum - p_low) / max(p_high - p_low, 1e-5), 0.0, 1.0)
        return np.where(mask, lum, 0.0)

    f_lum = get_luminance(front_up, f_mask)
    b_lum = get_luminance(back_up, b_mask)

    # En la cara trasera usamos directamente el esculpido real de la vista trasera
    # (sin estirar filas) donde b_mask está presente, con transición suave en bordes
    b_dist_in = ndimage.distance_transform_edt(b_mask).astype(np.float32)
    b_weight = np.clip(b_dist_in / (4.0 * scale), 0.0, 1.0)
    b_lum_combined = b_weight * b_lum + (1.0 - b_weight) * (0.75 * f_lum)
    b_lum_combined = np.where(f_mask, b_lum_combined, 0.0)

    # 1. Campo de volumen 3D anatómico (transformada de distancia multi-escala)
    dist = ndimage.distance_transform_edt(f_mask).astype(np.float32)
    max_r = 36.0 * scale
    d_clamped = np.minimum(dist, max_r)
    t_norm = d_clamped / max_r
    vol_profile = np.sqrt(np.clip(2.0 * t_norm - t_norm**2, 0.0, 1.0)) * (32.0 * scale)
    vol_broad = (
        ndimage.gaussian_filter(np.minimum(dist, 50.0 * scale), sigma=8.0 * scale)
        * 0.45
    )
    vol_smooth = ndimage.gaussian_filter(vol_profile + vol_broad, sigma=2.0 * scale)

    # En la zona de la peana (y >= 272 * scale), dar perfil cilíndrico 3D real
    yy, xx = np.mgrid[0:H, 0:W]
    base_y_start = 272 * scale
    base_transition = np.clip((yy - base_y_start) / (14.0 * scale), 0.0, 1.0)
    row_half_w = np.zeros((H, 1), dtype=np.float32)
    row_cx = np.full((H, 1), W / 2.0, dtype=np.float32)
    for y in range(H):
        xs_row = np.where(f_mask[y])[0]
        if len(xs_row) > 2:
            row_half_w[y, 0] = 0.5 * (xs_row[-1] - xs_row[0])
            row_cx[y, 0] = 0.5 * (xs_row[-1] + xs_row[0])
    cyl_z = np.sqrt(np.maximum(row_half_w**2 - (xx - row_cx) ** 2, 0.0))
    vol_combined = (1.0 - base_transition) * vol_smooth + base_transition * np.maximum(
        vol_smooth, cyl_z
    )

    # 2. Relieve de detalle (Shape-from-Shading multi-escala)
    def compute_sfs_detail(lum: np.ndarray) -> np.ndarray:
        low = ndimage.gaussian_filter(lum, sigma=7.0 * scale)
        mid = ndimage.gaussian_filter(lum, sigma=1.2 * scale) - low
        high = lum - ndimage.gaussian_filter(lum, sigma=1.4 * scale)
        return (0.55 * lum + 1.15 * mid + 0.75 * high) * (8.5 * scale)

    f_detail = compute_sfs_detail(f_lum)
    b_detail = compute_sfs_detail(b_lum_combined)

    edge_taper = np.clip(dist / (3.0 * scale), 0.0, 1.0)
    detail_taper = np.clip((dist - 1.5) / (4.0 * scale), 0.0, 1.0)
    z_front = np.where(
        f_mask,
        np.maximum(vol_combined * edge_taper + f_detail * detail_taper, 0.4),
        0.0,
    )
    z_back = np.where(
        f_mask,
        -np.maximum(vol_combined * edge_taper + b_detail * detail_taper, 0.4),
        0.0,
    )
    # Suavizado ligero en 2D para superficies limpias sin artefactos de borde
    z_front = np.where(f_mask, ndimage.gaussian_filter(z_front, sigma=0.65), 0.0)
    z_back = np.where(f_mask, ndimage.gaussian_filter(z_back, sigma=0.65), 0.0)

    # 3. Construcción de malla 100% cerrada (watertight) mediante celdas con paredes laterales
    # Eliminar posibles uniones diagonales de 1 solo vértice (non-manifold)
    valid_quads = (
        f_mask[:-1, :-1] & f_mask[1:, :-1] & f_mask[:-1, 1:] & f_mask[1:, 1:]
    )
    valid_quads = ndimage.binary_opening(valid_quads, structure=np.ones((2, 2)))
    lbl_q, num_q = ndimage.label(valid_quads)
    if num_q > 1:
        q_sizes = ndimage.sum(valid_quads, lbl_q, range(1, num_q + 1))
        valid_quads = lbl_q == (np.argmax(q_sizes) + 1)

    # Vértices en las esquinas de los quads activos
    corner_active = np.zeros((H, W), dtype=bool)
    corner_active[:-1, :-1] |= valid_quads
    corner_active[1:, :-1] |= valid_quads
    corner_active[:-1, 1:] |= valid_quads
    corner_active[1:, 1:] |= valid_quads

    idx_front = -np.ones((H, W), dtype=np.int32)
    idx_back = -np.ones((H, W), dtype=np.int32)

    ys_c, xs_c = np.where(corner_active)
    n_corners = len(ys_c)
    idx_front[ys_c, xs_c] = np.arange(n_corners, dtype=np.int32)
    idx_back[ys_c, xs_c] = np.arange(n_corners, 2 * n_corners, dtype=np.int32)

    verts_front = np.column_stack(
        [xs_c.astype(np.float64), (H - 1 - ys_c).astype(np.float64), z_front[ys_c, xs_c]]
    )
    verts_back = np.column_stack(
        [xs_c.astype(np.float64), (H - 1 - ys_c).astype(np.float64), z_back[ys_c, xs_c]]
    )
    verts_np = np.vstack([verts_front, verts_back])

    qy, qx = np.where(valid_quads)
    f00 = idx_front[qy, qx]
    f10 = idx_front[qy + 1, qx]
    f01 = idx_front[qy, qx + 1]
    f11 = idx_front[qy + 1, qx + 1]

    b00 = idx_back[qy, qx]
    b10 = idx_back[qy + 1, qx]
    b01 = idx_back[qy, qx + 1]
    b11 = idx_back[qy + 1, qx + 1]

    tris_list = [
        np.stack([f00, f10, f11], axis=1),
        np.stack([f00, f11, f01], axis=1),
        np.stack([b00, b11, b10], axis=1),
        np.stack([b00, b01, b11], axis=1),
    ]

    # Paredes laterales donde un quad activo limita con el exterior
    pad_q = np.pad(valid_quads, 1, mode="constant", constant_values=False)
    # Borde superior (y - 1 vacío): arista (00 -> 01)
    top_b = valid_quads & (~pad_q[:-2, 1:-1])
    ty, tx = np.where(top_b)
    tris_list.append(np.stack([idx_front[ty, tx], idx_front[ty, tx + 1], idx_back[ty, tx + 1]], axis=1))
    tris_list.append(np.stack([idx_front[ty, tx], idx_back[ty, tx + 1], idx_back[ty, tx]], axis=1))

    # Borde inferior (y + 1 vacío): arista (11 -> 10)
    bot_b = valid_quads & (~pad_q[2:, 1:-1])
    by, bx = np.where(bot_b)
    tris_list.append(np.stack([idx_front[by + 1, bx + 1], idx_front[by + 1, bx], idx_back[by + 1, bx]], axis=1))
    tris_list.append(np.stack([idx_front[by + 1, bx + 1], idx_back[by + 1, bx], idx_back[by + 1, bx + 1]], axis=1))

    # Borde izquierdo (x - 1 vacío): arista (10 -> 00)
    left_b = valid_quads & (~pad_q[1:-1, :-2])
    ly, lx = np.where(left_b)
    tris_list.append(np.stack([idx_front[ly + 1, lx], idx_front[ly, lx], idx_back[ly, lx]], axis=1))
    tris_list.append(np.stack([idx_front[ly + 1, lx], idx_back[ly, lx], idx_back[ly + 1, lx]], axis=1))

    # Borde derecho (x + 1 vacío): arista (01 -> 11)
    right_b = valid_quads & (~pad_q[1:-1, 2:])
    ry, rx = np.where(right_b)
    tris_list.append(np.stack([idx_front[ry, rx + 1], idx_front[ry + 1, rx + 1], idx_back[ry + 1, rx + 1]], axis=1))
    tris_list.append(np.stack([idx_front[ry, rx + 1], idx_back[ry + 1, rx + 1], idx_back[ry, rx + 1]], axis=1))

    all_tris = np.vstack(tris_list)
    mesh = trimesh.Trimesh(vertices=verts_np, faces=all_tris, process=True)

    # Suavizado Laplaciano suave en 3D para redondear el contorno lateral
    trimesh.smoothing.filter_laplacian(mesh, lamb=0.35, iterations=3)

    # Orientar para slicers 3D: X = ancho, Y = profundidad (-Z_img), Z = altura (Y_img)
    v = mesh.vertices.copy()
    mesh.vertices = np.column_stack([v[:, 0], -v[:, 2], v[:, 1]])

    # Apoyar en Z = 0 antes de medir la peana
    mesh.vertices[:, 2] -= mesh.bounds[0, 2]
    total_h_px = mesh.extents[2]

    # Seleccionar los vértices de la peana (10% inferior de la altura) y escalar a 30.0 x 30.0 mm
    base_mask = mesh.vertices[:, 2] <= (0.10 * total_h_px)
    base_verts = mesh.vertices[base_mask]
    base_w_px = float(base_verts[:, 0].max() - base_verts[:, 0].min())
    base_d_px = float(base_verts[:, 1].max() - base_verts[:, 1].min())

    scale_x = base_size_mm / base_w_px
    scale_y = base_size_mm / base_d_px
    scale_z = scale_x  # Altura proporcional al diámetro de la peana

    base_cx = 0.5 * float(base_verts[:, 0].max() + base_verts[:, 0].min())
    base_cy = 0.5 * float(base_verts[:, 1].max() + base_verts[:, 1].min())

    mesh.vertices[:, 0] = (mesh.vertices[:, 0] - base_cx) * scale_x
    mesh.vertices[:, 1] = (mesh.vertices[:, 1] - base_cy) * scale_y
    mesh.vertices[:, 2] = np.maximum(mesh.vertices[:, 2] * scale_z - 0.35, 0.0)

    # Verificar medidas finales de la peana
    final_base = mesh.vertices[mesh.vertices[:, 2] <= 4.0]
    peana_x = float(final_base[:, 0].max() - final_base[:, 0].min())
    peana_y = float(final_base[:, 1].max() - final_base[:, 1].min())

    print(
        f"[OK] Archivo STL generado en: {output_stl}\n"
        f"     - Vértices: {len(mesh.vertices):,}\n"
        f"     - Triángulos: {len(mesh.faces):,}\n"
        f"     - Es malla cerrada (watertight): {mesh.is_watertight}\n"
        f"     - Dimensiones de la peana (mm): {peana_x:.1f} x {peana_y:.1f} mm\n"
        f"     - Dimensiones totales (mm): {mesh.extents[0]:.1f} (ancho) x {mesh.extents[1]:.1f} (fondo) x {mesh.extents[2]:.1f} (alto) mm"
    )
    mesh.export(output_stl)
    return mesh


def render_stl_preview(mesh: trimesh.Trimesh, preview_path: Path) -> None:
    """
    Renderiza una lámina con 4 vistas del archivo STL (Frontal, 3/4, Trasera, Lateral)
    usando z-buffer por píxel e iluminación de estudio tipo resina.
    """
    v = mesh.vertices
    f = mesh.faces
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(fn, axis=1, keepdims=True)
    fn = fn / np.maximum(norms, 1e-8)
    tri_centers = (v0 + v1 + v2) / 3.0

    views = [
        ("Vista Frontal (STL)", 0.0),
        ("Vista 3/4 (STL)", np.deg2rad(32.0)),
        ("Vista Trasera (STL)", np.pi),
        ("Vista Perfil (STL)", np.deg2rad(90.0)),
    ]

    panel_w, panel_h = 360, 390
    canvas = np.full((panel_h, panel_w * len(views), 3), 242, dtype=np.uint8)

    light1 = np.array([-0.35, -0.82, 0.45])
    light1 /= np.linalg.norm(light1)
    light2 = np.array([0.45, -0.60, -0.25])
    light2 /= np.linalg.norm(light2)

    scale_px = 7.4

    for idx, (_title, yaw) in enumerate(views):
        cos_a, sin_a = np.cos(yaw), np.sin(yaw)
        rot = np.array(
            [
                [cos_a, -sin_a, 0.0],
                [sin_a, cos_a, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        tc_r = tri_centers @ rot.T
        fn_r = fn @ rot.T

        vis = fn_r[:, 1] <= 0.05
        tc_v = tc_r[vis]
        fn_v = fn_r[vis]

        order = np.argsort(-tc_v[:, 1])
        tc_v = tc_v[order]
        fn_v = fn_v[order]

        diff1 = np.clip(np.sum(fn_v * light1, axis=1), 0.0, 1.0)
        diff2 = np.clip(np.sum(fn_v * light2, axis=1), 0.0, 1.0)
        view_dir = np.array([0.0, -1.0, 0.0])
        half_vec = light1 + view_dir
        half_vec /= np.linalg.norm(half_vec)
        spec = np.clip(np.sum(fn_v * half_vec, axis=1), 0.0, 1.0) ** 24

        intensity = 0.24 + 0.62 * diff1 + 0.18 * diff2
        base_col = np.array([108.0, 166.0, 198.0])
        rgb = intensity[:, None] * base_col + spec[:, None] * 65.0
        cols = np.clip(rgb, 25, 250).astype(np.uint8)

        px = np.round(panel_w // 2 + tc_v[:, 0] * scale_px).astype(np.int32)
        py = np.round(panel_h - 32 - tc_v[:, 2] * scale_px).astype(np.int32)

        in_bounds = (px >= 1) & (px < panel_w - 1) & (py >= 1) & (py < panel_h - 1)
        px, py, cols = px[in_bounds], py[in_bounds], cols[in_bounds]

        panel = np.full((panel_h, panel_w, 3), 242, dtype=np.uint8)
        for dy in (0, 1):
            for dx in (0, 1):
                panel[py + dy, px + dx] = cols

        x_off = idx * panel_w
        canvas[:, x_off : x_off + panel_w] = panel

    img_pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img_pil)
    for idx, (title, _) in enumerate(views):
        draw.text((idx * panel_w + 115, 10), title, fill=(50, 60, 70))

    img_pil.save(preview_path)
    print(f"[OK] Previsualización del STL guardada en: {preview_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera un modelo 3D (.stl) de Elektra a partir de su imagen frontal y trasera."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Ruta a figurita.jpg con las vistas frontal y trasera",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"C:\git\test\elektra.stl"),
        help="Ruta de salida para el archivo .STL",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Token opcional de Hugging Face para la API cloud",
    )
    parser.add_argument(
        "--tripo-key",
        type=str,
        default=None,
        help="Clave API de Tripo3D (tsk_...). Si se omite, se busca TRIPO_API_KEY en .env",
    )
    parser.add_argument(
        "--from-mesh",
        type=Path,
        default=None,
        help="Convierte un archivo .glb/.obj descargado de studio.tripo3d.ai o makerworld.com a .STL",
    )
    parser.add_argument(
        "--fuse-meshes",
        nargs=2,
        type=Path,
        metavar=("FRONT_GLB", "BACK_GLB"),
        default=None,
        help="Fusiona 2 modelos .glb gratuitos de 1 sola vista (frontal y trasera) en un único .STL 30x30 mm",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Omitir intento cloud (útil bajo proxy corporativo Zscaler) y generar directamente en local",
    )
    args = parser.parse_args()

    out_dir = args.output.parent
    front_png = out_dir / "elektra_frontal.png"
    back_png = out_dir / "elektra_trasera.png"
    preview_png = out_dir / "elektra_stl_preview.png"

    if args.fuse_meshes:
        mesh = fuse_front_back_meshes(args.fuse_meshes[0], args.fuse_meshes[1], args.output)
        render_stl_preview(mesh, preview_png)
        return

    if args.from_mesh:
        mesh = convert_mesh_to_stl(args.from_mesh, args.output)
        render_stl_preview(mesh, preview_png)
        return

    front_rgba, back_rgba = extract_clean_views(args.input, front_png, back_png)

    tripo_key = args.tripo_key or load_env_key("TRIPO_API_KEY")
    if not args.local_only and tripo_key:
        mesh_tripo = try_tripo3d_multiview(front_png, back_png, args.output, tripo_key)
        if mesh_tripo is not None:
            render_stl_preview(mesh_tripo, preview_png)
            return

    if not args.local_only:
        if try_huggingface_multiview(front_png, back_png, args.output, args.hf_token):
            return
        print(
            "[Info] No se pudo usar la API cloud (sin créditos API o bloqueo Zscaler).\n"
            "       Activando reconstrucción 3D bilateral local..."
        )

    mesh = build_bilateral_3d_stl(front_rgba, back_rgba, args.output)
    render_stl_preview(mesh, preview_png)


if __name__ == "__main__":
    main()
