"""Create the first GeoFusion project progress report from EDA outputs."""

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "GeoFusion_Initial_Dataset_and_EDA_Report.docx"
SUMMARY = ROOT / "data" / "eda" / "recording_summary.csv"


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    tc_pr.append(shading)


def set_cell_border(cell, color="A6A6A6") -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = "w:" + edge
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "4")
        element.set(qn("w:color"), color)


def set_run_font(run, bold=False, size=11):
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    run._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor(0, 0, 0)


def add_paragraph(doc, text="", bold_lead=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.15
    if bold_lead and text.startswith(bold_lead):
        set_run_font(p.add_run(bold_lead), bold=True)
        set_run_font(p.add_run(text[len(bold_lead):]))
    else:
        set_run_font(p.add_run(text))
    return p


def add_heading(doc, text, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    p.paragraph_format.space_before = Pt(12 if level == 1 else 8)
    p.paragraph_format.space_after = Pt(5)
    run = p.add_run(text)
    set_run_font(run, bold=True, size=12 if level == 1 else 11)
    return p


def add_table(doc, headers, rows, widths=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    table.autofit = False
    for i, header in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        set_cell_shading(cell, "D9D9D9")
        set_cell_border(cell)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_run_font(p.add_run(str(header)), bold=True)
        if widths:
            cell.width = Cm(widths[i])
    for row_index, row in enumerate(rows):
        cells = table.add_row().cells
        for i, value in enumerate(row):
            cell = cells[i]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_border(cell)
            if row_index % 2 == 1:
                set_cell_shading(cell, "F2F2F2")
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.space_before = Pt(2)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i == 0 and len(str(value)) < 35 else WD_ALIGN_PARAGRAPH.LEFT
            set_run_font(p.add_run(str(value)))
            if widths:
                cell.width = Cm(widths[i])
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return table


def add_figure(doc, path: Path, caption: str, interpretation: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(3)
    run = p.add_run()
    run.add_picture(str(path), width=Cm(15.8))
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c.paragraph_format.space_after = Pt(4)
    set_run_font(c.add_run(caption), bold=True)
    add_paragraph(doc, interpretation, bold_lead="Interpretation: ")


def set_document_defaults(doc):
    section = doc.sections[0]
    section.top_margin = Cm(1.8)
    section.bottom_margin = Cm(1.8)
    section.left_margin = Cm(1.8)
    section.right_margin = Cm(1.8)
    styles = doc.styles
    for style_name in ["Normal", "Title", "Heading 1", "Heading 2"]:
        style = styles[style_name]
        style.font.name = "Times New Roman"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
        style.font.color.rgb = RGBColor(0, 0, 0)
    styles["Normal"].font.size = Pt(11)


def main():
    OUT.parent.mkdir(exist_ok=True)
    s = pd.read_csv(SUMMARY)
    normal = s[s["class"] == "Normal road"]
    anomaly = s[s["class"] == "Road anomaly"]
    total_rows = int(s["rows"].sum())
    valid_duration = s["duration_seconds"].dropna()
    doc = Document()
    set_document_defaults(doc)

    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(6)
    set_run_font(title.add_run("GeoFusion Initial Dataset and Exploratory Data Analysis Report"), bold=True, size=11)
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(14)
    set_run_font(subtitle.add_run("Multimodal Road Condition Intelligence and GIS Based Maintenance Prioritization"), size=11)

    add_paragraph(doc, "This report records the initial working scope for GeoFusion after inspection of the selected road-sensing dataset. It covers the resources required for development, principal risks, the available data and modalities, exploratory analysis, and the path toward a deployable smartphone application.")

    add_heading(doc, "1 Project Scope and Current Data Position")
    add_paragraph(doc, "GeoFusion is intended to convert mobile road observations into road-anomaly assessments and GIS-based maintenance priorities. The final system objective remains a mobile workflow in which a vehicle-mounted smartphone captures road conditions, records inertial motion and location, produces an anomaly or severity estimate, and maps the result to a road segment for maintenance decision support.")
    add_paragraph(doc, "The selected dataset provides synchronized smartphone-sensor, location, elevation and weather fields. Its raw-data component, retained externally because of storage constraints, also contains annotation files, camera folders and text files pointing to the corresponding Google Drive videos. The initial implementation can therefore use IMU and GIS data immediately while treating video as an externally linked visual modality that must be downloaded or streamed only when required for frame-level analysis.")

    add_heading(doc, "2 Description of the Selected Dataset")
    add_paragraph(doc, "The supplied archive contains event-level combined CSV recordings with GIS and weather fields, isolated sensor recordings, and a GIS package containing route elevation data. The Raw Data component is available separately at E:\\downloads_2\\Raw Data and is accessed in place. The completed EDA uses the combined CSV recordings for synchronized sensor analysis and the raw annotation files for event-level analysis.")
    add_table(doc,
        ["Dataset characteristic", "Observed value"],
        [
            ["Analysed recordings", "102 CSV recordings"],
            ["Raw session folders available", "102 folders in the supplied Raw Data copy"],
            ["Sensor observations", f"{total_rows:,} rows"],
            ["Fields per combined recording", "45"],
            ["Road-anomaly recordings", "97"],
            ["Normal-road recordings", "5"],
            ["Median sampling rate", "100 Hz"],
            ["Median recording duration", f"{valid_duration.median():.2f} seconds"],
            ["Observed latitude range", "24.361684 to 24.532970"],
            ["Observed longitude range", "88.522954 to 88.839153"],
            ["Auxiliary GIS file", "Route elevation CSV with latitude, longitude and elevation"],
            ["External video links found", "21 Google Drive link files in the available Raw Data copy"],
        ], widths=[7.2, 9.5])
    add_paragraph(doc, "Each combined event file includes elapsed time; calibrated and uncalibrated accelerometer and gyroscope values; gravity, orientation, magnetometer, compass and barometer readings; GPS latitude and longitude; GIS elevation; and temperature, humidity, cloud cover and wind speed. The class assignment used in this report is derived from the source archive folders, not from the largely empty annotation-text column.")

    add_heading(doc, "3 Modalities Selected for the Initial Model")
    add_table(doc,
        ["Modality", "Available fields", "Role in GeoFusion"],
        [
            ["Inertial sensing", "Three-axis total acceleration and gyroscope signals at 100 Hz", "Primary signal for road-impact and vibration pattern detection."],
            ["Geospatial context", "GPS latitude, GPS longitude and GIS elevation", "Location, map matching, elevation context and later road-segment aggregation."],
            ["Weather context", "Temperature, humidity, cloud cover and wind speed", "Contextual features for robustness analysis; not treated as the primary anomaly signal."],
            ["Visual road imagery", "Google Drive links referenced by session camera folders; videos are not stored locally", "Visual validation and later RGB plus IMU fusion after selected videos are accessed and temporally aligned."],
        ], widths=[3.3, 5.6, 7.8])
    add_paragraph(doc, "For the present dataset, the working multimodal formulation is inertial sensing plus geospatial context, with weather retained as auxiliary context. This is a valid heterogeneous fusion setup because the IMU measures vehicle motion while the GIS fields describe physical location and terrain context. Video is now confirmed as an available but externally linked modality. It should be incorporated only after the required recordings and their timestamp annotations have been verified.")

    add_heading(doc, "4 Exploratory Data Analysis")
    add_paragraph(doc, "Data exploration was completed at two levels. Record-level profiling measured file count, row count, duration, estimated sample rate, missingness and representative geographic position. Sensor-level profiling calculated the magnitude of total acceleration and gyroscope vectors to inspect the scale and variability of the inertial signals. GPS coverage and contextual GIS and weather fields were also checked.")
    add_table(doc,
        ["Measure", "Road anomaly recordings", "Normal-road recordings"],
        [
            ["Number of recordings", "97", "5"],
            ["Median rows per recording", f"{anomaly['rows'].median():,.0f}", f"{normal['rows'].median():,.0f}"],
            ["Median duration", f"{anomaly['duration_seconds'].median():.2f} s", f"{normal['duration_seconds'].median():.2f} s"],
            ["Median estimated sample rate", f"{anomaly['estimated_sampling_hz'].median():.0f} Hz", f"{normal['estimated_sampling_hz'].median():.0f} Hz"],
            ["Mean acceleration magnitude", f"{anomaly['acceleration_magnitude_mean'].mean():.3f} m/s²", f"{normal['acceleration_magnitude_mean'].mean():.3f} m/s²"],
            ["Mean gyroscope magnitude", f"{anomaly['gyroscope_magnitude_mean'].mean():.3f} rad/s", f"{normal['gyroscope_magnitude_mean'].mean():.3f} rad/s"],
            ["Median missing-cell percentage", f"{anomaly['missing_cell_percent'].median():.2f}%", f"{normal['missing_cell_percent'].median():.2f}%"],
        ], widths=[6.5, 5.1, 5.1])
    add_paragraph(doc, "The recordings exhibit a stable median sampling rate of 100 Hz, which is suitable for window-based time-series analysis. Acceleration magnitude is close to gravitational acceleration in both classes because total acceleration includes the gravity component. The relevant discrimination signal will therefore come from variation, peaks, spectral energy and orientation-normalized motion features rather than the raw mean alone.")
    add_paragraph(doc, "The strongest data issue is class imbalance: 97 anomaly recordings versus 5 normal recordings, or approximately 19.4 to 1. The initial anomaly model must therefore avoid reporting ordinary accuracy as the main result. Stratified evaluation where feasible, class-weighted loss, precision-recall analysis, balanced accuracy and F1-score will be used. The normal class should also be expanded through further collection before any deployment claim is made.")
    add_paragraph(doc, "A data-quality screen also identified one anomaly file with only four rows and no usable duration. This file should be excluded from window generation unless the raw acquisition can be recovered. GPS latitude or longitude is absent at the record-summary level in a small number of files, and these cases should be handled through missing-location flags rather than fabricated coordinates.")
    add_table(doc,
        ["Raw annotation result", "Observed value"],
        [
            ["Valid annotated events", "688"],
            ["Bump annotations", "396"],
            ["Pothole annotations", "292"],
            ["Sessions containing bump or pothole annotations", "97"],
            ["Median bump press duration", "83.5 ms"],
            ["Median pothole press duration", "83.0 ms"],
        ], widths=[8.5, 8.2])
    add_paragraph(doc, "The raw annotations provide the event type and elapsed time needed to create properly labelled sensor windows. The annotation button-press duration is retained as metadata; it is not interpreted as physical road-defect severity.")

    add_figure(doc, ROOT / "data" / "eda" / "recording_overview.png", "Figure 1. Recording count and recording-duration distribution by source class.", "Interpretation: The record-level class split is heavily skewed toward anomaly-labelled sessions. Recording duration also varies substantially, particularly for anomalies, so the modelling unit should be fixed-duration windows rather than whole files.")
    add_figure(doc, ROOT / "data" / "eda" / "acceleration_distribution.png", "Figure 2. Total-acceleration magnitude distribution by source class.", "Interpretation: Both distributions are centred near gravitational acceleration because total acceleration includes gravity. The visible overlap shows that a raw magnitude average is not sufficient for classification. Window-level peaks, variance, jerk, spectral energy and orientation-aware features should be evaluated.")
    add_figure(doc, ROOT / "data" / "eda" / "recording_locations.png", "Figure 3. Median GPS position for each recording with valid coordinates.", "Interpretation: The observations cover multiple route locations, confirming the usefulness of geospatial context. Median positions are suitable for a first coverage check, but deployment should use the complete GPS trace and map matching rather than one representative coordinate per session.")
    add_figure(doc, ROOT / "data" / "eda" / "annotation_event_types.png", "Figure 4. Counts of bump and pothole annotations from the raw session files.", "Interpretation: The raw annotations provide two explicit anomaly categories, with more bump annotations than pothole annotations. This supports a later three-class formulation of normal road, bump and pothole, provided normal windows are expanded and split at the session level.")
    add_figure(doc, ROOT / "data" / "eda" / "annotation_press_duration.png", "Figure 5. Distribution of annotation button-press duration by anomaly type.", "Interpretation: The two event types have very similar median annotation durations, around 83 milliseconds. This field describes the manual annotation action rather than defect intensity, so it will be used for data-quality review and not as a predictive feature or severity target.")

    add_heading(doc, "5 Video Data Exploration and Sensor Video Alignment")
    add_paragraph(doc, "The Raw Data folders were inspected in place at E:\\downloads_2\\Raw Data. This check found 81 local MP4 files in session camera folders. Each video has a corresponding session-level Annotation.csv file, and all 402 bump or pothole annotations in these video-backed sessions fall within the duration of their respective videos. A lightweight manifest was generated to store video paths, event type, sensor timestamp, target video frame and video metadata without copying the MP4 files into the project workspace.")
    add_table(doc,
        ["Video EDA measure", "Observed value"],
        [
            ["Local MP4 recordings", "81"],
            ["Video-backed annotated events", "402"],
            ["Video-backed bump events", "232"],
            ["Video-backed pothole events", "170"],
            ["Events within video bounds", "402 of 402"],
            ["Video resolution", "4K; 3840 x 2160 or 2160 x 3840"],
            ["Orientation by session", "56 portrait and 25 landscape"],
            ["Observed frame-rate range", "27.49 to 30.02 frames per second"],
            ["Frame-quality audit", "16 balanced annotated frames read successfully"],
        ], widths=[8.0, 8.7])
    add_paragraph(doc, "The video filenames encode their recording-start epoch in milliseconds. Cross-checking this value with the annotation timestamp and seconds_elapsed field confirmed that the event time can be converted into a target frame. This establishes the essential data link for the proposed multimodal pipeline: a short visual clip centred on an event can be paired with an IMU window centred on the same event.")
    add_figure(doc, ROOT / "data" / "eda" / "video_inventory_eda.png", "Figure 6. Video-backed event count and recording orientation across the available local MP4 sessions.", "Interpretation: The local video subset contains both anomaly types, but bump events are more frequent. Most recordings are portrait rather than landscape, so the visual preprocessing pipeline must detect orientation and rotate frames consistently before image-model training.")
    add_figure(doc, ROOT / "data" / "eda" / "video_frame_quality_eda.png", "Figure 7. Brightness and sharpness measures for a balanced 16-frame visual-audit sample.", "Interpretation: These measures are a data-quality check, not a classification result. The sample demonstrates that labelled frames can be decoded and quantified directly from the MP4 files. Brightness and sharpness vary across driving conditions, so colour normalization and image-quality augmentation should be included in the visual branch.")
    add_figure(doc, ROOT / "data" / "eda" / "video_event_contact_sheet.png", "Figure 8. Sample frames extracted at annotation-aligned bump and pothole timestamps.", "Interpretation: The contact sheet verifies that the project has usable road-facing visual evidence linked to the sensor labels. Road anomalies may be small, partially occluded or not visually prominent in one isolated frame. Accordingly, the model should use a short clip or several neighbouring frames together with the IMU window, rather than assuming that every event can be identified from a single image.")

    add_heading(doc, "6 Resource Allocation")
    add_table(doc,
        ["Work package", "Resources", "Deliverable"],
        [
            ["Data preparation", "Python, pandas, NumPy; local project storage; version-controlled preprocessing scripts", "Clean event windows, quality flags, class labels and train-validation-test splits."],
            ["EDA and feature engineering", "Matplotlib or Seaborn; IMU time-series utilities", "Signal magnitude, peak, variance, frequency-domain and GPS-derived features."],
            ["Model development", "PyTorch or TensorFlow; CPU for baselines and optional GPU for deep time-series models", "IMU baseline, GIS-context fusion model and calibrated evaluation."],
            ["GIS and decision support", "GeoPandas or QGIS; map data and road-segment layer", "Map matching, segment aggregation and maintenance-priority score."],
            ["Mobile deployment", "Android smartphone with camera, IMU and GPS; TensorFlow Lite or ONNX Runtime Mobile", "On-device sensing and inference prototype with map-oriented output."],
        ], widths=[3.2, 7.0, 6.5])
    add_paragraph(doc, "The current data volume is manageable on a standard development laptop. Training can start with engineered time-window features and lightweight one-dimensional CNN or temporal models. This keeps the early project stage reproducible and avoids allocating cloud resources before a valid baseline and test protocol are in place.")

    add_heading(doc, "7 Risk Analysis and Mitigation")
    add_table(doc,
        ["Risk", "Effect on project", "Mitigation"],
        [
            ["Severe class imbalance", "A model may learn to label most windows as anomalies or may fail to represent normal driving.", "Collect or obtain additional normal-road recordings; use class weights; report F1, balanced accuracy and precision-recall measures."],
            ["Externally linked video availability", "Video is not stored locally and the supplied Raw Data copy contains only a subset of session link files.", "Use the derived link index to select and verify videos; retain sensor-only baselines; document the exact sessions for which video is accessible."],
            ["Short or incomplete recordings", "Unreliable time windows and distorted feature distributions.", "Apply minimum-duration checks, remove the four-row file from model input, and retain an auditable exclusion log."],
            ["GPS gaps or low location precision", "Incorrect map matching and road-segment prioritization.", "Use accuracy fields and missing-coordinate flags; perform map matching only when location quality is sufficient."],
            ["Device and vehicle variation", "A model may not generalize across phones, mounts, vehicles or speeds.", "Normalize signals, capture metadata, use augmentation and evaluate by route, device and vehicle where possible."],
            ["Mobile compute and battery constraints", "A large model may be unsuitable for real-time operation.", "Use windowed inference, a lightweight model, quantization and periodic rather than continuous map uploads."],
        ], widths=[3.8, 6.2, 6.7])

    add_heading(doc, "8 Immediate Development Plan and Mobile Deployment Path")
    add_paragraph(doc, "The next development step is to convert each recording into fixed-duration windows, compute quality flags and create a reproducible split that prevents closely related windows from leaking across train and test partitions. A simple IMU baseline will then be compared with an IMU plus GIS and weather-context fusion model. The output will be an anomaly probability and confidence score rather than an unsupported maintenance decision.")
    add_paragraph(doc, "For deployment, the smartphone application will collect camera frames, accelerometer, gyroscope and GPS readings. In the first mobile version, the trained IMU plus GIS model will run locally to flag probable road anomalies. GPS coordinates will be stored only for accepted events and uploaded for map matching and road-segment aggregation. The external linked videos will support visual validation and development of the visual branch. The resulting dashboard will rank road segments by repeated confirmed events, estimated severity and location context.")

    add_heading(doc, "9 Conclusion")
    add_paragraph(doc, "The selected dataset is suitable for starting GeoFusion because it provides high-frequency inertial signals together with GPS, elevation, weather context and externally linked videos across the available sessions. The EDA has confirmed a stable nominal sampling rate, usable geospatial coverage and 688 raw event annotations, while also identifying a strong shortage of normal-road examples and incomplete availability of local video links. The appropriate immediate scope is therefore an IMU plus GIS anomaly-analysis prototype, followed by targeted video retrieval and timestamp alignment for the complete mobile GeoFusion system.")

    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
