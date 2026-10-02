import os
import tempfile
import gradio as gr

# Hugging Face ZeroGPU compatibility
try:
    import spaces
    GPU_DECORATOR = spaces.GPU
except ImportError:
    def GPU_DECORATOR(duration=120):
        def decorator(fn):
            return fn
        return decorator

from src.pipeline.multicam_pipeline import MultiCamTrackingPipeline

pipeline_instance = None


def get_pipeline():
    global pipeline_instance
    if pipeline_instance is None:
        pipeline_instance = MultiCamTrackingPipeline()
    return pipeline_instance


@GPU_DECORATOR(duration=120)
def process_tracking_ui(video_files, model_name, tracker_type, reid_thresh):
    """Process uploaded videos with tracking and cross-camera ReID."""
    if not video_files:
        return None, "Please upload at least one video file."

    if not isinstance(video_files, list):
        video_files = [video_files]

    video_paths = []
    for vf in video_files:
        if isinstance(vf, str):
            video_paths.append(vf)
        elif hasattr(vf, "name"):
            video_paths.append(vf.name)

    if not video_paths:
        return None, "Invalid video files provided."

    temp_out_dir = tempfile.mkdtemp()

    pipe = get_pipeline()
    pipe.config["detector"]["model_name"] = model_name
    pipe.config["tracker"]["tracker_type"] = tracker_type
    pipe.config["reid"]["reid_threshold"] = float(reid_thresh)

    try:
        output_files = pipe.process_videos(video_paths, output_dir=temp_out_dir)

        if output_files:
            combined = [f for f in output_files if "combined" in os.path.basename(f)]
            main_output = combined[0] if combined else output_files[0]
            status_msg = (
                "Processing completed.\n"
                f"Videos processed : {len(video_paths)}\n"
                f"Detector model   : {model_name}\n"
                f"Tracker          : {tracker_type.upper()}\n"
                f"Output files     : {len(output_files)}"
            )
            return main_output, status_msg
        return None, "Error: no output videos were generated."
    except Exception as e:
        return None, f"Pipeline error: {str(e)}"


custom_css = """

:root {
    --c-bg: #FAF9F5;
    --c-surface: #FFFFFF;
    --c-inset: #F5F4ED;
    --c-border: #E6E3D8;
    --c-border-strong: #D6D2C4;
    --c-text: #1F1E1D;
    --c-muted: #6F6D66;
    --c-accent: #C6613F;
    --c-accent-hover: #AE5330;
    --c-accent-soft: #F6E4DB;
    --c-on-accent: #FFFFFF;

    --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    --font-serif: ui-serif, Georgia, Cambria, "Times New Roman", serif;
}

.dark {
    --c-bg: #262624;
    --c-surface: #30302E;
    --c-inset: #1F1E1D;
    --c-border: #434340;
    --c-border-strong: #55544F;
    --c-text: #FAF9F5;
    --c-muted: #A9A79D;
    --c-accent: #D97757;
    --c-accent-hover: #E48A6B;
    --c-accent-soft: #4A3329;
    --c-on-accent: #FFFFFF;
}

.gradio-container.gradio-container {
    --font: var(--font-sans);
    --body-background-fill: var(--c-bg);
    --background-fill-primary: var(--c-surface);
    --background-fill-secondary: var(--c-inset);
    --body-text-color: var(--c-text);
    --body-text-color-subdued: var(--c-muted);

    --block-background-fill: var(--c-surface);
    --block-border-color: var(--c-border);
    --block-border-width: 1px;
    --block-shadow: none;
    --block-radius: 12px;

    --border-color-primary: var(--c-border);
    --border-color-accent: var(--c-accent);
    --color-accent: var(--c-accent);
    --color-accent-soft: var(--c-accent-soft);
    --link-text-color: var(--c-accent);
    --link-text-color-hover: var(--c-accent-hover);
    --link-text-color-active: var(--c-accent-hover);
    --slider-color: var(--c-accent);

    --input-background-fill: var(--c-inset);
    --input-background-fill-hover: var(--c-inset);
    --input-background-fill-focus: var(--c-inset);
    --input-border-color: var(--c-border);
    --input-border-color-hover: var(--c-border-strong);
    --input-border-color-focus: var(--c-accent);
    --input-border-width: 1px;
    --input-radius: 8px;
    --input-shadow: none;
    --input-shadow-focus: 0 0 0 3px var(--c-accent-soft);
    --input-text-color: var(--c-text);

    --button-primary-background-fill: var(--c-accent);
    --button-primary-background-fill-hover: var(--c-accent-hover);
    --button-primary-text-color: var(--c-on-accent);
    --button-primary-border-color: var(--c-accent);
    --button-secondary-background-fill: var(--c-surface);
    --button-secondary-background-fill-hover: var(--c-inset);
    --button-secondary-text-color: var(--c-text);
    --button-secondary-border-color: var(--c-border-strong);

    /* Labels and titles: no background, no border */
    --block-label-background-fill: transparent;
    --block-label-border-width: 0px;
    --block-label-border-color: transparent;
    --block-label-shadow: none;
    --block-label-radius: 0;
    --block-label-text-color: var(--c-text);
    --block-title-background-fill: transparent;
    --block-title-border-width: 0px;
    --block-title-padding: 0;
    --block-title-radius: 0;
    --block-title-text-color: var(--c-text);
}

html { overflow-y: scroll; }   /* keeps width stable when scrollbar would appear */

body,
.gradio-container.gradio-container {
    background: var(--c-bg) !important;
    color: var(--c-text);
    font-family: var(--font-sans) !important;
}

.gradio-container.gradio-container {
    width: 100% !important;
    max-width: none !important;
    margin: 0 !important;
    padding: 0 !important;
}

.gradio-container .app,
.gradio-container main,
.gradio-container .main,
.gradio-container .fillable {
    width: 100% !important;
    max-width: 1560px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    box-sizing: border-box !important;
}

.gradio-container .app,
.gradio-container .main {
    padding: 28px 24px 40px !important;
}

footer { display: none !important; }


.app-header {
    background: none !important;
    border: 0 !important;
    padding: 0 2px 6px !important;
}
.app-header h1 {
    font-family: var(--font-serif) !important;
    font-size: clamp(1.15rem, 1.5vw, 1.5rem) !important;
    font-weight: 500 !important;
    letter-spacing: -0.01em !important;
    line-height: 1.3 !important;
    color: var(--c-text) !important;
    text-align: left !important;
    margin: 0 !important;
}


/* Dropdown, slider, textbox titles */
.gradio-container [data-testid="block-info"] {
    background: none !important;
    border: 0 !important;
    box-shadow: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
    margin: 0 0 6px 0 !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    line-height: 1.4 !important;
    color: var(--c-text) !important;
}

/* File / video labels (normally a floating chip) */
.gradio-container [data-testid="block-label"] {
    position: static !important;
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
    box-sizing: border-box !important;
    background: none !important;
    border: 0 !important;
    box-shadow: none !important;
    border-radius: 0 !important;
    margin: 0 !important;
    padding: 14px 16px 8px !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    line-height: 1.4 !important;
    color: var(--c-text) !important;
}
.gradio-container [data-testid="block-label"] svg,
.gradio-container [data-testid="block-label"] span:has(> svg) {
    display: none !important;
}

/* Accordion title */
.gradio-container .label-wrap,
.gradio-container .label-wrap span {
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    color: var(--c-text) !important;
}

.demo-card {
    background: var(--c-surface) !important;
    border: 1px solid var(--c-border) !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
    margin-bottom: 18px !important;
    align-items: center !important;
    gap: 16px !important;
}
.demo-card-info h4 {
    margin: 0 0 2px 0 !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: var(--c-text) !important;
}
.demo-card-info p {
    margin: 0 !important;
    font-size: 0.85rem !important;
    color: var(--c-muted) !important;
}


.gradio-container button[role="tab"] {
    font-size: 0.925rem !important;
    font-weight: 500 !important;
    color: var(--c-muted) !important;
    background: none !important;
    border-radius: 0 !important;
}
.gradio-container button[role="tab"].selected,
.gradio-container button[role="tab"][aria-selected="true"] {
    color: var(--c-text) !important;
}

.gradio-container button.primary,
.gradio-container button.secondary {
    min-height: 42px !important;
    border-radius: 8px !important;
    font-size: 0.925rem !important;
    font-weight: 500 !important;
    box-shadow: none !important;
    transition: background-color 0.15s ease, border-color 0.15s ease !important;
}
.gradio-container button.primary {
    background: var(--c-accent) !important;
    border: 1px solid var(--c-accent) !important;
    color: var(--c-on-accent) !important;
}
.gradio-container button.primary:hover {
    background: var(--c-accent-hover) !important;
    border-color: var(--c-accent-hover) !important;
}
.gradio-container button.secondary {
    background: var(--c-surface) !important;
    border: 1px solid var(--c-border-strong) !important;
    color: var(--c-text) !important;
}
.gradio-container button.secondary:hover {
    background: var(--c-inset) !important;
}

.main-row {
    flex-wrap: nowrap !important;
    gap: 20px !important;
    align-items: flex-start !important;
}
.col-left  { flex: 5 1 0% !important; min-width: 0 !important; }
.col-right { flex: 7 1 0% !important; min-width: 0 !important; }

.video-box { min-height: 340px; }
.video-box video {
    width: 100% !important;
    height: auto !important;
    max-height: 520px;
    object-fit: contain;
}
.log-box textarea {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important;
    font-size: 0.85rem !important;
    line-height: 1.5 !important;
}


.how-it-works {
    max-width: 820px;
    color: var(--c-text);
}
.how-it-works h3 {
    font-family: var(--font-serif) !important;
    font-weight: 500 !important;
    font-size: 1.25rem !important;
    color: var(--c-text) !important;
}
.how-it-works li, .how-it-works p {
    color: var(--c-text) !important;
    line-height: 1.65 !important;
}

@media (max-width: 1024px) {
    .gradio-container .app,
    .gradio-container .main { padding: 20px 16px 32px !important; }
    .main-row { gap: 16px !important; }
    .video-box { min-height: 300px; }
    .video-box video { max-height: 420px; }
}

@media (max-width: 820px) {
    .gradio-container .app,
    .gradio-container .main { padding: 14px 12px 28px !important; }

    .main-row,
    .demo-card {
        flex-direction: column !important;
        flex-wrap: wrap !important;
    }
    .col-left,
    .col-right,
    .demo-card > div {
        flex: 1 1 100% !important;
        width: 100% !important;
        min-width: 0 !important;
    }

    .demo-card { padding: 12px 14px !important; gap: 12px !important; }
    .gradio-container button.primary,
    .gradio-container button.secondary { width: 100% !important; }

    .video-box { min-height: 220px; }
    .video-box video { max-height: 300px; }
    .log-box textarea { font-size: 0.8rem !important; }
}

@media (max-width: 480px) {
    .app-header h1 { font-size: 1.05rem !important; }
    .demo-card-info h4 { font-size: 0.9rem !important; }
    .demo-card-info p { font-size: 0.8rem !important; }
}
"""

# Warm terracotta ramp used as the primary hue
terracotta = gr.themes.Color(
    c50="#FBF3EF", c100="#F6E3DA", c200="#EDC8B8", c300="#E3A78D",
    c400="#D98A69", c500="#D07350", c600="#C6613F", c700="#A94E31",
    c800="#8A4029", c900="#6F3524", c950="#3E1D13",
)

theme = gr.themes.Base(
    primary_hue=terracotta,
    secondary_hue=terracotta,
    neutral_hue=gr.themes.colors.stone,
)


def get_demo_videos():
    candidates = ["videos/init/Double1.mp4", "videos/init/Single1.mp4"]
    return [p for p in candidates if os.path.exists(p)]


with gr.Blocks(
    css=custom_css,
    theme=theme,
    title="Multi-Camera Person Tracking and Re-Identification",
    fill_width=True,
) as demo:

    gr.HTML(
        """
        <div class="app-header">
            <h1>Multi-Camera Person Tracking and Re-Identification</h1>
        </div>
        """
    )

    sample_files = get_demo_videos()
    with gr.Row(
        visible=len(sample_files) > 0,
        elem_classes=["demo-card"],
    ) as demo_card_row:
        with gr.Column(scale=4, min_width=0):
            gr.HTML(
                """
                <div class="demo-card-info">
                    <h4>Demo videos</h4>
                    <p>Load the sample dual-camera videos to test tracking and cross-camera ReID.</p>
                </div>
                """
            )
        with gr.Column(scale=1, min_width=0):
            btn_load_demo = gr.Button("Load Demo Videos", variant="secondary", size="md")

    with gr.Tabs():
        with gr.TabItem("Tracking Studio"):
            with gr.Row(elem_classes=["main-row"]):
                with gr.Column(scale=5, min_width=0, elem_classes=["col-left"]):
                    input_videos = gr.File(
                        label="Input videos",
                        file_count="multiple",
                        file_types=["video"],
                    )

                    with gr.Accordion("Settings", open=True):
                        model_dropdown = gr.Dropdown(
                            choices=[
                                "yolov8n.pt", "yolov8s.pt", "yolov8m.pt",
                                "yolov8x.pt", "yolo11n.pt", "yolo11m.pt",
                            ],
                            value="yolov8m.pt",
                            label="Detector model",
                        )
                        tracker_dropdown = gr.Dropdown(
                            choices=["bytetrack", "botsort"],
                            value="bytetrack",
                            label="Tracker",
                        )
                        reid_slider = gr.Slider(
                            minimum=0.10,
                            maximum=0.50,
                            value=0.25,
                            step=0.01,
                            label="ReID threshold",
                            info="Cosine distance. Lower is stricter.",
                        )

                    btn_run = gr.Button("Run", variant="primary", size="lg")

                with gr.Column(scale=7, min_width=0, elem_classes=["col-right"]):
                    output_video = gr.Video(
                        label="Output video",
                        autoplay=True,
                        elem_classes=["video-box"],
                    )
                    status_output = gr.Textbox(
                        label="Log",
                        interactive=False,
                        lines=5,
                        elem_classes=["log-box"],
                    )

            def handle_load_demo():
                return get_demo_videos(), gr.update(visible=False)

            btn_load_demo.click(
                fn=handle_load_demo,
                inputs=[],
                outputs=[input_videos, demo_card_row],
            )

            btn_run.click(
                fn=process_tracking_ui,
                inputs=[input_videos, model_dropdown, tracker_dropdown, reid_slider],
                outputs=[output_video, status_output],
            )

        with gr.TabItem("How It Works"):
            gr.Markdown(
                """
                ### Pipeline

                1. **Detection** - YOLOv8 / YOLO11 pretrained weights detect people (COCO class 0) in each frame.
                2. **Tracking** - ByteTrack or BoT-SORT links detections across frames, using a Kalman filter to keep IDs stable through short occlusions.
                3. **Re-identification** - An OSNet backbone extracts L2-normalized 512-dimensional features from each person crop. Cosine distance between cameras is used to assign one global ID per person.
                4. **Interpolation** - Short gaps in each track are filled by linear interpolation to reduce box flicker.
                """,
                elem_classes=["how-it-works"],
            )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)