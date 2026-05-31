from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from contextlib import contextmanager

import streamlit as st
from PIL import Image, ImageOps

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_LOCAL_FOLDER = Path("/home/patricio/pics")
DEFAULT_TARGET_HEIGHT = 960
DEFAULT_OUTPUT_FORMAT = "WEBP"
DEFAULT_QUALITY = 90
DEFAULT_DELETE_ORIGINALS = True
DEFAULT_MAX_IMAGE_PIXELS = Image.MAX_IMAGE_PIXELS
OUTPUT_FORMATS = {
    "WEBP": {"extension": ".webp"},
    "PNG": {"extension": ".png"},
    "JPEG": {"extension": ".jpg"},
}


@dataclass
class ImageSource:
    name: str
    data: bytes
    original_path: Path | None = None


@contextmanager
def allow_large_image_load() -> Iterable[None]:
    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        yield
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit


def open_image(source: ImageSource) -> Image.Image:
    with allow_large_image_load():
        return Image.open(io.BytesIO(source.data))


@st.cache_data(show_spinner=False)
def get_image_dimensions(data: bytes) -> tuple[int, int]:
    with allow_large_image_load():
        with Image.open(io.BytesIO(data)) as image:
            return image.size


def find_giant_images(sources: list[ImageSource]) -> list[tuple[str, tuple[int, int], int]]:
    if DEFAULT_MAX_IMAGE_PIXELS is None:
        return []

    giant_images: list[tuple[str, tuple[int, int], int]] = []
    for source in sources:
        width, height = get_image_dimensions(source.data)
        total_pixels = width * height
        if total_pixels > DEFAULT_MAX_IMAGE_PIXELS:
            giant_images.append((source.name, (width, height), total_pixels))
    return giant_images


def choose_folder() -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory()
        root.destroy()
        return selected or None
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_images_from_folder(folder: str) -> list[ImageSource]:
    folder_path = Path(folder).expanduser()
    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=True)
    if not folder_path.is_dir():
        raise ValueError("This folder does not exist or is not valid.")

    sources: list[ImageSource] = []
    for file_path in sorted(folder_path.iterdir()):
        if file_path.suffix.lower() not in ALLOWED_EXTENSIONS or not file_path.is_file():
            continue
        sources.append(
            ImageSource(
                name=file_path.name,
                data=file_path.read_bytes(),
                original_path=file_path,
            )
        )

    if not sources:
        raise ValueError("No PNG, JPG, or WEBP images were found in the folder.")
    return sources


def load_uploaded_images(uploaded_files: Iterable[st.runtime.uploaded_file_manager.UploadedFile]) -> list[ImageSource]:
    sources: list[ImageSource] = []
    for uploaded_file in uploaded_files:
        extension = Path(uploaded_file.name).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            continue
        sources.append(ImageSource(name=uploaded_file.name, data=uploaded_file.getvalue()))
    return sources


def resize_image(image: Image.Image, target_height: int) -> Image.Image:
    normalized = ImageOps.exif_transpose(image)
    if normalized.mode not in {"RGB", "RGBA"}:
        if "A" in normalized.getbands():
            normalized = normalized.convert("RGBA")
        else:
            normalized = normalized.convert("RGB")

    scale_ratio = target_height / normalized.height
    target_width = max(1, round(normalized.width * scale_ratio))
    return normalized.resize((target_width, target_height), Image.Resampling.LANCZOS)


def save_image(
    image: Image.Image,
    destination: Path,
    output_format: str,
    quality: int,
) -> None:
    encoded = encode_image_bytes(image, output_format, quality)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(encoded)


def encode_image_bytes(image: Image.Image, output_format: str, quality: int) -> bytes:
    format_name = output_format.upper()
    params: dict[str, object] = {}

    if format_name == "PNG":
        params = {"optimize": True, "compress_level": 9}
        image_to_save = image if image.mode in {"RGB", "RGBA"} else image.convert("RGBA")
    elif format_name == "JPEG":
        params = {
            "quality": quality,
            "optimize": True,
            "progressive": True,
            "subsampling": "4:2:0",
        }
        image_to_save = image.convert("RGB")
    elif format_name == "WEBP":
        params = {
            "quality": quality,
            "method": 6,
        }
        image_to_save = image.convert("RGB") if image.mode == "RGBA" else image
    else:
        raise ValueError("Unsupported output format.")

    buffer = io.BytesIO()
    image_to_save.save(buffer, format=format_name, **params)
    return buffer.getvalue()


def build_destination(name: str, output_format: str, output_dir: Path) -> Path:
    stem = Path(name).stem
    extension = OUTPUT_FORMATS[output_format]["extension"]
    return output_dir / f"{stem}{extension}"


def ensure_available_destination(destination: Path) -> Path:
    candidate = destination
    counter = 1
    while candidate.exists():
        candidate = destination.with_name(f"{destination.stem}_procesada_{counter}{destination.suffix}")
        counter += 1
    return candidate


def resolve_destination(
    source: ImageSource,
    output_format: str,
    output_dir: Path,
    delete_originals: bool,
) -> tuple[Path, bool]:
    destination = build_destination(source.name, output_format, output_dir)

    if source.original_path is None:
        return ensure_available_destination(destination), False

    original_path = source.original_path.expanduser().resolve()
    same_destination = destination.resolve() == original_path
    if same_destination:
        if delete_originals:
            return destination, True
        return ensure_available_destination(destination), False

    return destination, False


def process_images(
    sources: list[ImageSource],
    target_height: int,
    output_format: str,
    quality: int,
    delete_originals: bool,
    output_dir: Path,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[list[Path], list[str]]:
    generated: list[Path] = []
    deleted: list[str] = []
    total_sources = len(sources)

    file_prefix = st.session_state.get("file_prefix", "")
    for index, source in enumerate(sources, start=1):
        if file_prefix:
            extension = OUTPUT_FORMATS[output_format]["extension"]
            destination = output_dir / f"{file_prefix}{index}{extension}"
            replaced_original = False
        else:
            destination, replaced_original = resolve_destination(
                source,
                output_format,
                output_dir,
                delete_originals,
            )

        with open_image(source) as image:
            resized = resize_image(image, target_height)
            save_image(resized, destination, output_format, quality)
            generated.append(destination)

        if delete_originals and source.original_path and source.original_path.exists() and not replaced_original:
            source.original_path.unlink()
            deleted.append(source.original_path.name)
        elif delete_originals and replaced_original and source.original_path:
            deleted.append(source.original_path.name)

        if progress_callback:
            progress_callback(index, total_sources, source.name)

    return generated, deleted


def generate_preview(source: ImageSource, target_height: int, output_format: str, quality: int) -> tuple[bytes, tuple[int, int], int]:
    with open_image(source) as image:
        resized = resize_image(image, target_height)
        dimensions = resized.size
        encoded = encode_image_bytes(resized, output_format, quality)
        return encoded, dimensions, len(encoded)


def quality_help_text(output_format: str) -> str:
    if output_format == "PNG":
        return "PNG  uses lossless compression with maximum automatic optimization."
    if output_format == "JPEG":
        return "JPEG uses lossy compression. High quality recommended between 85 and 92."
    return "WEBP balances size and quality. A range of 80 to 90 is usually the best compromise."


def format_file_size(size_in_bytes: int) -> str:
    units = ["B", "KB", "MB"]
    size = float(size_in_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size_in_bytes} B"


def format_pixels(total_pixels: int) -> str:
    return f"{total_pixels:,}".replace(",", ".")


def initialize_session_state() -> None:
    defaults = {
        "target_height": DEFAULT_TARGET_HEIGHT,
        "output_format": DEFAULT_OUTPUT_FORMAT,
        "quality": DEFAULT_QUALITY,
        "delete_originals": DEFAULT_DELETE_ORIGINALS,
        "selected_folder": str(DEFAULT_LOCAL_FOLDER),
        "folder_path": str(DEFAULT_LOCAL_FOLDER),
        "output_folder": str(DEFAULT_LOCAL_FOLDER),
        "upload_uploader_key": 0,
        "preview_image": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_app_state() -> None:
    st.session_state.target_height = DEFAULT_TARGET_HEIGHT
    st.session_state.output_format = DEFAULT_OUTPUT_FORMAT
    st.session_state.quality = DEFAULT_QUALITY
    st.session_state.delete_originals = DEFAULT_DELETE_ORIGINALS
    st.session_state.selected_folder = str(DEFAULT_LOCAL_FOLDER)
    st.session_state.folder_path = str(DEFAULT_LOCAL_FOLDER)
    st.session_state.output_folder = str(DEFAULT_LOCAL_FOLDER)
    st.session_state.preview_image = None
    st.session_state.upload_uploader_key += 1


def show_processing_feedback() -> None:
    result = st.session_state.pop("processing_result", None)
    if not result:
        return

    st.success(
        f"Process completed. {result['generated_count']} images were generated in '{result['output_dir']}'."
    )
    if result["deleted"]:
        st.warning(f"Originals deleted: {', '.join(result['deleted'])}")


def main() -> None:
    st.set_page_config(page_title="PicsManager", page_icon="🖼️", layout="wide")
    st.title("PicsManager")
    st.write("Resize and compress images in batch from uploaded files or a local folder.")

    initialize_session_state()
    show_processing_feedback()

    with st.sidebar:
        st.header("Configuration")
        if st.button("Clear", use_container_width=True):
            reset_app_state()
            st.rerun()

        target_height = st.number_input(
            "Target Height (px)",
            min_value=100,
            max_value=8000,
            key="target_height",
            step=10,
        )
        output_format = st.selectbox(
            "Output Format",
            options=list(OUTPUT_FORMATS.keys()),
            key="output_format",
        )
        quality = st.selectbox(
            "Quality",
            options=list(range(80, 96)),
            key="quality",
            disabled=output_format == "WEBP",
        )
        delete_originals = st.checkbox(
            "Delete originals from folder",
            key="delete_originals",
            help="Only applies to images read from a local folder.",
        )
        st.text_input(
            "Output Folder",
            key="output_folder",
            placeholder=str(DEFAULT_LOCAL_FOLDER),
            help="Folder where the processed images will be saved.",
        )
        st.text_input(
            "File Name Prefix",
            key="file_prefix",
            placeholder="prefix_",
            help="Prefix that will be added to the names of the processed files.",
        )

        st.caption(quality_help_text(output_format))

    source_tab, upload_tab = st.tabs(["Local Folder", "Upload Images"])

    folder_sources: list[ImageSource] = []
    upload_sources: list[ImageSource] = []

    with source_tab:
        col1, col2 = st.columns([3, 1])
        with col1:
            folder_path = st.text_input(
                "Folder Path",
                key="folder_path",
                placeholder=str(DEFAULT_LOCAL_FOLDER),
            )
        with col2:
            st.write("")
            if st.button("Select Folder"):
                selected = choose_folder()
                if selected:
                    previous_source = st.session_state.selected_folder
                    st.session_state.selected_folder = selected
                    st.session_state.folder_path = selected
                    if st.session_state.output_folder == previous_source:
                        st.session_state.output_folder = selected
                    st.rerun()

        if folder_path:
            st.session_state.selected_folder = folder_path
            try:
                folder_sources = load_images_from_folder(folder_path)
                st.success(f"Loaded {len(folder_sources)} images from the folder.")
            except ValueError as error:
                st.error(str(error))

    with upload_tab:
        uploaded_files = st.file_uploader(
            "Select one or more images",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key=f"uploaded_files_{st.session_state.upload_uploader_key}",
        )
        if uploaded_files:
            upload_sources = load_uploaded_images(uploaded_files)
            st.success(f"Loaded {len(upload_sources)} images from the browser.")

    output_dir = Path(st.session_state.output_folder).expanduser()
    sources = folder_sources or upload_sources
    if sources:
        preview_options = [source.name for source in sources[:10]]
        if st.session_state.preview_image not in preview_options:
            st.session_state.preview_image = preview_options[0]

        preview_names = ", ".join(source.name for source in sources[:5])
        extra = "" if len(sources) <= 5 else f" and {len(sources) - 5} more"
        st.info(f"Images ready for processing: {preview_names}{extra}")

        giant_images = find_giant_images(sources)
        if giant_images:
            warning_lines = []
            for name, dimensions, total_pixels in giant_images[:3]:
                warning_lines.append(
                    f"- {name}: {dimensions[0]} x {dimensions[1]} ({format_pixels(total_pixels)} pixels)"
                )
            remaining = len(giant_images) - len(warning_lines)
            suffix = ""
            if remaining > 0:
                suffix = f"\n- and {remaining} more giant images"
            st.warning(
                "Giant images were detected that exceed the default Pillow safety threshold. "
                "The app will process them anyway, but they may consume a lot of memory.\n"
                + "\n".join(warning_lines)
                + suffix
            )

        st.subheader("Preview before saving")
        selected_name = st.selectbox(
            "Image to compare",
            options=preview_options,
            help="Up to 10 images are shown for quick preview.",
            key="preview_image",
        )
        selected_source = next(
            (source for source in sources if source.name == selected_name),
            sources[0],
        )
        preview_bytes, preview_dimensions, preview_size = generate_preview(
            selected_source,
            int(target_height),
            output_format,
            int(quality),
        )

        col_original, col_preview = st.columns(2)
        with col_original:
            st.caption(
                f"Original: {selected_name} | {format_file_size(len(selected_source.data))}"
            )
            st.image(selected_source.data, width='stretch')
        with col_preview:
            st.caption(
                f"Resultado: {preview_dimensions[0]} x {preview_dimensions[1]} | {output_format} | {format_file_size(preview_size)}"
            )
            st.image(preview_bytes, width='stretch')

    if st.button("Process Images", type="primary", disabled=not sources):
        progress_bar = st.progress(0, text="Preparing processing...")
        progress_status = st.empty()

        def update_progress(current: int, total: int, name: str) -> None:
            progress_bar.progress(current / total, text=f"Processing {current}/{total}: {name}")
            progress_status.caption(f"Last processed image: {name}")

        try:
            generated, deleted = process_images(
                sources=sources,
                target_height=int(target_height),
                output_format=output_format,
                quality=int(quality),
                delete_originals=delete_originals,
                output_dir=output_dir,
                progress_callback=update_progress,
            )
        except Exception as error:
            progress_bar.empty()
            progress_status.empty()
            st.exception(error)
        else:
            progress_bar.progress(1.0, text="Processing completed")
            load_images_from_folder.clear()
            st.session_state.processing_result = {
                "generated_count": len(generated),
                "deleted": deleted,
                "output_dir": str(output_dir),
            }
            st.rerun()


if __name__ == "__main__":
    main()
