"""Rebuild the GeoFusion progress report while retaining the supplied cover."""
from pathlib import Path
from math import atan2, cos, sin
from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import black, HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

ROOT=Path(__file__).resolve().parents[1]; REPORTS=ROOT/'reports'
COVER=Path(r'E:\downloads_2\GeoFusion_Initial_Dataset_and_EDA_Report.pdf')
BODY=REPORTS/'_geofusion_progress_body.pdf'; FINAL=REPORTS/'GeoFusion_Progress_Report_Novelty_Methodology_Results.pdf'

def init():
    pdfmetrics.registerFont(TTFont('TNR',r'C:\Windows\Fonts\times.ttf'))
    pdfmetrics.registerFont(TTFont('TNRB',r'C:\Windows\Fonts\timesbd.ttf'))
    b=getSampleStyleSheet()['BodyText']
    return {'b':ParagraphStyle('b',parent=b,fontName='TNR',fontSize=10,leading=12.9,alignment=TA_JUSTIFY,spaceAfter=6),
            's':ParagraphStyle('s',parent=b,fontName='TNR',fontSize=8.55,leading=10.3,alignment=TA_JUSTIFY),
            'h':ParagraphStyle('h',parent=b,fontName='TNRB',fontSize=8.55,leading=10.3),
            'd':ParagraphStyle('d',parent=b,fontName='TNR',fontSize=7.35,leading=8.25,alignment=TA_CENTER),
            'db':ParagraphStyle('db',parent=b,fontName='TNRB',fontSize=8.35,leading=9.2,alignment=TA_CENTER)}

def p(c,t,style,x,y,w):
    q=Paragraph(t,style); _,z=q.wrap(w,1000); q.drawOn(c,x,y-z); return y-z
def foot(c,n):
    c.setStrokeColor(black);c.setLineWidth(.3);c.line(2*cm,1.5*cm,19*cm,1.5*cm)
    c.setFont('TNR',8.3);c.drawString(2*cm,1.03*cm,'GeoFusion Progress Report');c.drawRightString(19*cm,1.03*cm,f'Page {n}')
def start(c,title,n):
    c.setFillColor(black);c.setFont('TNRB',15);c.drawString(2*cm,27.2*cm,title);foot(c,n);return 26.4*cm
def tbl(c,rows,widths,y,s):
    data=[[Paragraph(str(v),s['h'] if i==0 else s['s']) for v in r] for i,r in enumerate(rows)]
    q=Table(data,colWidths=widths,repeatRows=1)
    q.setStyle(TableStyle([('LINEABOVE',(0,0),(-1,0),.6,black),('LINEBELOW',(0,0),(-1,0),.8,black),('LINEBELOW',(0,1),(-1,-2),.2,HexColor('#777777')),('LINEBELOW',(0,-1),(-1,-1),.5,black),('LINEAFTER',(0,0),(-2,-1),.2,black),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
    _,h=q.wrap(sum(widths),700);q.drawOn(c,2*cm,y-h);return y-h

def arr(c,x1,y1,x2,y2,color,dash=False):
    c.setStrokeColor(color);c.setLineWidth(1.05);c.setDash(3,2) if dash else c.setDash();c.line(x1,y1,x2,y2)
    a=atan2(y2-y1,x2-x1);r=4.5
    c.line(x2,y2,x2-r*cos(a-.5),y2-r*sin(a-.5));c.line(x2,y2,x2-r*cos(a+.5),y2-r*sin(a+.5));c.setDash()
def box(c,x,y,w,h,title,text,fill,edge,s):
    c.setFillColor(fill);c.setStrokeColor(edge);c.setLineWidth(.75);c.roundRect(x,y,w,h,5,1,1)
    c.setFillColor(edge);p(c,title,s['db'],x+4,y+h-10,w-8);c.setFillColor(black);p(c,text,s['d'],x+4,y+8,w-8)
def architecture(c,s):
    blue,orange,green,purp,grey=[HexColor(v) for v in ('#1769AA','#C45B11','#287B49','#6B4C9A','#5E6670')]
    bf,of,gf,pf,xg=[HexColor(v) for v in ('#E7F1FA','#FFF0E3','#E8F5EC','#F0EAF8','#F1F3F5')]
    # Input modalities
    box(c,2*cm,19.9*cm,4.1*cm,1.85*cm,'Camera stream','Road-facing MP4; t = 0 frame and compact frames at -1, 0, +1 seconds.',bf,blue,s)
    box(c,6.85*cm,19.9*cm,4.1*cm,1.85*cm,'Phone sensor stream','Accelerometer, gyroscope, gravity and sampling-time readings in a 3-second window.',of,orange,s)
    box(c,11.7*cm,19.9*cm,3.35*cm,1.85*cm,'Spatial stream','GPS coordinate, timestamp and session identity retained with each event.',gf,green,s)
    box(c,15.8*cm,19.9*cm,3.2*cm,1.85*cm,'Annotations','Bump/Pothole time anchor; separated windows form weak negatives.',xg,grey,s)
    # preparation
    box(c,2*cm,16.55*cm,4.1*cm,1.85*cm,'Visual preparation','Time alignment, frame resizing, clip decode check and visual sample construction.',bf,blue,s)
    box(c,6.85*cm,16.55*cm,4.1*cm,1.85*cm,'Inertial features','RMS/deviation, peak acceleration, jerk, gyroscope peak and sampling rate.',of,orange,s)
    box(c,12.0*cm,16.55*cm,5.0*cm,1.85*cm,'Split and quality control','Fixed session-level 70/15/15 split; no journey shared between subsets.',xg,grey,s)
    # encoders
    box(c,2*cm,12.8*cm,4.1*cm,2.05*cm,'YOLO11n classifier','224-pixel visual event classifier producing p<sub>vision</sub>; no boxes needed.',bf,blue,s)
    box(c,6.85*cm,12.8*cm,4.1*cm,2.05*cm,'IMU logistic model','Physical-response classifier producing p<sub>IMU</sub>.',of,orange,s)
    box(c,12.0*cm,12.8*cm,5.0*cm,2.05*cm,'Early fusion comparator','3D clip embedding plus IMU features before classification; evaluated, not selected.',pf,purp,s)
    # output
    box(c,5.0*cm,8.9*cm,6.0*cm,2.15*cm,'Selected late fusion','p<sub>event</sub> = 0.5p<sub>vision</sub> + 0.5p<sub>IMU</sub>; selection by validation F1.',pf,purp,s)
    box(c,12.65*cm,8.9*cm,4.35*cm,2.15*cm,'Event record','Fused confidence, event type, GPS, time and provisional impact priority.',gf,green,s)
    box(c,6.45*cm,4.85*cm,8.0*cm,2.15*cm,'GIS maintenance layer','Map matching; 10-25 m road segments; recurrence, recency, confidence and impact priority; MapLibre and QGIS.',gf,green,s)
    # arrows
    arr(c,4.05*cm,19.9*cm,4.05*cm,18.4*cm,blue);arr(c,8.9*cm,19.9*cm,8.9*cm,18.4*cm,orange)
    arr(c,17.4*cm,19.9*cm,14.6*cm,18.4*cm,grey,True);arr(c,4.05*cm,16.55*cm,4.05*cm,14.85*cm,blue);arr(c,8.9*cm,16.55*cm,8.9*cm,14.85*cm,orange)
    arr(c,4.05*cm,16.55*cm,14.5*cm,14.85*cm,purp,True);arr(c,8.9*cm,16.55*cm,14.5*cm,14.85*cm,purp,True)
    arr(c,4.05*cm,12.8*cm,7.1*cm,11.05*cm,purp);arr(c,8.9*cm,12.8*cm,8.8*cm,11.05*cm,purp);arr(c,11*cm,10*cm,12.65*cm,10*cm,green);arr(c,14.8*cm,8.9*cm,10.6*cm,7*cm,green)
    c.setFillColor(grey);c.setFont('TNR',7.3);c.drawString(2*cm,4.15*cm,'Solid arrows: selected inference path. Dashed arrows: comparison and supervision paths.')
    p(c,'Figure 1. GeoFusion multimodal architecture implemented to date. Distinct modality representations are preserved until late fusion, while spatial fields remain available for map audit and road-segment aggregation.',s['s'],2*cm,3.7*cm,17*cm)

def make():
    s=init();REPORTS.mkdir(exist_ok=True);c=canvas.Canvas(str(BODY),pagesize=A4)
    # 2
    y=start(c,'1. Project novelty and research framing',2)
    for t in [
      'GeoFusion treats road-condition intelligence as a sensing, inference and decision problem rather than an isolated pothole-image classifier. It links road-facing video, phone inertial response and GPS observations so that a detected event becomes a traceable map record. The intended output is therefore not merely an image label: it retains location, confidence, physical-response evidence and a route to maintenance prioritisation.',
      'The research contribution is the explicit comparison of unimodal, early-fusion and late-fusion evidence on aligned RGB-IMU windows. Visual evidence helps when surface appearance is visible; inertial evidence helps when lighting, blur or visual ambiguity limits the camera. GeoFusion preserves both probabilities rather than hiding one modality inside a single opaque feature vector. A reviewer can inspect whether a report was driven by the camera, the sensor response, or their agreement.',
      'The project separates <b>event identity</b>, <b>sensor impact priority</b> and <b>maintenance priority</b>. The annotations identify Bump and Pothole events but do not provide inspected mild, moderate or severe defect labels. The present Low, Medium and High Impact tiers are consequently relative signals calculated from phone response. They are suitable for ranking and triage, not as a measurement of pothole depth, road roughness or civil-engineering severity.',
      'A GIS-first record design is the third contribution. Each inference retains time, latitude, longitude, visual probability, inertial probability, fused probability and impact priority. Repeated reports can be aggregated at road-segment level rather than displayed as uncontrolled point alerts. This supplies an auditable transition from mobile sensing to analyst-facing maintenance information.']:
        y=p(c,t,s['b'],2*cm,y,17*cm)
    tbl(c,[['Research element','Implemented contribution and boundary'],['Multimodal event unit','A video event anchor is paired with a 3-second inertial window and GPS record; it is not a single independently sampled image.'],['Fusion study','IMU baseline, HOG, YOLO11n, compact clip encoder, early fusion and late fusion are compared under one session split.'],['Deployment traceability','Separate probabilities and derived impact indicators remain available for map audit and later threshold review.'],['Evaluation discipline','Session-level separation and validation-only selection reduce leakage; the enhanced comparison has not used the held-out test set.']],[5*cm,12*cm],y-3,s);c.showPage()
    # 3
    y=start(c,'2. Methodology: data preparation and event construction',3)
    for t in [
      'The experiments use RoadSens-4M, a phone-sensing road dataset combining inertial, orientation, GPS/GIS, weather and camera information. The processed experimental collection contains 102 sensor recordings, 616,870 sensor rows and 45 fields. It includes 97 anomaly recordings and 5 normal-road recordings. The median sampling rate is approximately 100 Hz and the median recording duration is 29.09 seconds, establishing the temporal resolution available for short event-centred windows.',
      'Raw camera annotations were inspected before modelling. Eighty-one locally accessible MP4 sessions contain 402 annotations aligned to video: 231 Bump and 171 Pothole events. Alignment uses the recording start epoch with annotation seconds elapsed; every aligned target frame lies inside the recorded video interval. Seven clips could not be decoded and were excluded with their failures logged, leaving 641 prepared compact clips for learned clip experiments.']:
        y=p(c,t,s['b'],2*cm,y,17*cm)
    y=tbl(c,[['Preparation stage','Procedure and resulting dataset'],['Time alignment','Convert annotation seconds to camera time and sensor timestamp; preserve source session identity.'],['Event window','Extract sensor observations from 1.5 seconds before to 1.5 seconds after each annotation.'],['Candidate-normal sampling','329 windows sampled away from known annotations after quality checks; they are weak negatives, not certified smooth-road labels.'],['Visual samples','Event-centred frame for HOG and YOLO; frames at -1, 0 and +1 seconds for compact clips.'],['Spatial recovery','Latitude and longitude recovered for all 402 locally aligned events and retained with timestamp.']],[5*cm,12*cm],y-3,s)
    y=p(c,'Sensor features are calculated over each three-second interval. They include sampling rate, acceleration RMS and deviation from local baseline, acceleration peak, jerk and gyroscope peak. These descriptors form the inertial baseline input and generate the provisional impact score. The score is normalised over the supplied local event set so that tiers retain a consistent relative interpretation during this first phase.',s['b'],2*cm,y-7,17*cm)
    y=p(c,'Quality control excludes windows with unavailable aligned video, failed clip decoding, insufficient observations for a complete sensor interval or missing essential time fields. The annotation and recording identifiers are retained through each transformation. This makes it possible to trace a model input, an event probability and a future GIS record back to the underlying journey and event anchor.',s['b'],2*cm,y,17*cm)
    p(c,'The final event unit is therefore deliberately multimodal and temporally bounded. It combines visual context, the local motion signature and a spatial observation from the same road passage. This construction avoids treating a single image as an independent description of a road defect, while also avoiding the false precision of assigning engineering severity labels that are absent from the source annotations.',s['b'],2*cm,y,17*cm);c.showPage()
    # 4
    y=start(c,'Methodology continued: multimodal model architecture',4)
    p(c,'Figure 1 expresses the implemented architecture as interacting modality paths, not a simple serial workflow. Camera, sensor, annotation and spatial information retain distinct roles during preparation. The selected path classifies visual and inertial evidence independently and combines their probabilities late; the early-fusion clip branch remains a controlled comparator.',s['b'],2*cm,y,17*cm)
    architecture(c,s);c.showPage()
    # 5
    y=start(c,'Learning and evaluation protocol',5)
    for t in [
      'The initial task is deliberately narrow: predict an annotated road event versus a temporally separated candidate-normal window. It measures discrimination within supplied journeys and does not yet claim universal road-quality recognition. This establishes a defensible multimodal benchmark while the project develops a reviewed, inspection-based severity dataset for later work.',
      'A deterministic 70/15/15 split is fixed at session level before model comparison. A recording appears in training, validation or test only once, preventing adjacent windows from one journey being shared across subsets. The usable validation subset contains 109 windows, including 51 positive events. The held-out test subset contains 81 windows, including 45 positives, and remains untouched after the enhanced visual and fusion comparison.']:
        y=p(c,t,s['b'],2*cm,y,17*cm)
    y=tbl(c,[['Model family','Input representation','Role in study'],['IMU logistic baseline','Sampling rate, RMS/deviation and peak acceleration, jerk, gyroscope peak.','Physical-response reference.'],['HOG frame baseline','One event-centred video frame with hand-engineered gradients.','Classical visual reference.'],['YOLO11n classification','224-pixel event-centred frame; image-level classification, not box detection.','Learned visual probability.'],['Compact 3D clip encoder','Three 96-pixel frames at -1, 0 and +1 seconds.','Short-motion visual representation.'],['Early fusion','Clip embedding concatenated with IMU vector before classification.','Feature-level comparator.'],['Late fusion','Equal-weight average of YOLO and IMU probabilities.','Selected probability-level path.']],[3.6*cm,7.1*cm,6.3*cm],y-3,s)
    y=p(c,'All preliminary selection uses validation F1 at a probability threshold of 0.50, while average precision and ROC-AUC check whether the ranking is useful beyond one threshold. The late-fusion weight is fixed at 0.5 per modality before scoring rather than fitting a validation meta-classifier. The compact clip result is interpreted alongside precision and ranking metrics because high recall alone can be produced by indiscriminate positive calls.',s['b'],2*cm,y-7,17*cm)
    y=p(c,'The visual experiments progressively increase representation capacity: hand-engineered HOG features provide a low-complexity reference; YOLO11n evaluates learned single-frame evidence; and the compact 3D encoder examines whether a small temporal clip improves discrimination. The fusion experiments then ask a separate question: whether combining a learned temporal visual representation with sensor features before classification is more reliable than combining independently calibrated modality probabilities after classification.',s['b'],2*cm,y,17*cm)
    p(c,'This sequence is methodologically useful because it separates improvement caused by a stronger vision encoder from improvement caused by a fusion rule. It also preserves a practical deployment comparison: the selected late-fusion rule can continue to operate if one modality yields a low-confidence signal, while still recording both inputs for later review.',s['b'],2*cm,y,17*cm);c.showPage()
    # 6
    y=start(c,'3. First-level results and model selection',6)
    y=p(c,'The values below are validation results on the same 109 usable windows. They are first-level experimental results, not final deployment estimates. The comparison identifies the model to take forward without spending the held-out test set during iteration.',s['b'],2*cm,y,17*cm)
    y=tbl(c,[['Model','Precision','Recall','F1','Average precision','Reading of result'],['IMU-only','0.700','0.412','0.519','0.723','Strong ranking; conservative calls at 0.50.'],['HOG frame','0.468','0.569','0.513','0.524','Weak visual generalisation.'],['YOLO11n classification','0.561','0.627','0.593','0.619','Best learned frame-level visual result.'],['Compact clip encoder','0.468','1.000','0.638','0.588','Near-universal positive calls inflate F1.'],['Early clip + IMU','0.571','0.549','0.560','0.723','No validated feature-level gain.'],['Late YOLO + IMU','0.612','0.588','0.600','0.704','Selected method by validation F1.'],['Late clip + IMU','0.710','0.431','0.537','0.725','Good ranking but insufficient recall.']],[3.1*cm,1.8*cm,1.55*cm,1.3*cm,2.05*cm,7.2*cm],y-4,s)
    for t in ['Late YOLO plus IMU fusion is selected because its validation F1 of 0.600 exceeds early clip-plus-IMU fusion (0.560), YOLO-only (0.593) and HOG (0.513). Its operational benefit is equally important: p<sub>vision</sub> and p<sub>IMU</sub> can both be stored with the fused decision, so an analyst can distinguish modality agreement from a decision dominated by one stream.', 'The clip encoder illustrates why a threshold metric cannot be read alone. Its recall of 1.000 is accompanied by precision equal to positive prevalence, showing that it flags nearly every validation window. Average precision of 0.588 and ROC-AUC of 0.565 indicate limited discrimination. This negative result prevents an inappropriate selection based only on the largest displayed F1.', 'The comparison also explains why early fusion was not selected at this stage. Its F1 of 0.560 is lower than late YOLO plus IMU fusion, despite using a learned clip embedding and the same inertial evidence. With the present sample size, retaining modality-specific representations until the probability stage produces a simpler and more interpretable decision rule. This result will be revisited only after the final test evaluation and collection of reviewed field labels.']:
        y=p(c,t,s['b'],2*cm,y-8,17*cm)
    c.showPage()
    # 7
    y=start(c,'First-level results continued: interpretation and limitations',7)
    for t in [
      'The modality comparison supports the central premise that the two streams contribute different evidence. The IMU model gives the strongest average precision among single-stream models, indicating useful ranking from the phone physical response. YOLO improves recall at the selected threshold and supplies visual context. Late fusion improves the balance of precision and recall without requiring the high-dimensional clip-plus-sensor representation used by early fusion.',
      'Several limitations define the scope of the conclusion. Candidate-normal windows are sampled away from annotations rather than independently verified as smooth road segments. The target distinguishes recorded anomaly events from these candidate-normal windows, not every type of road damage. The class distribution is also imbalanced at recording level, with 97 anomaly and 5 normal-road sessions; session-aware validation and calibration remain essential as data collection grows.',
      'Video availability constrains the paired study. The local visual analysis uses 81 MP4 sessions and 402 aligned annotations, whereas the wider sensor collection contains 102 recordings. Decode failures are logged rather than silently replaced. These choices preserve traceability, but the present outcome must be interpreted as a paired-modality feasibility study rather than final evidence for every source recording.']:
        y=p(c,t,s['b'],2*cm,y,17*cm)
    y=tbl(c,[['Current claim','Supported interpretation','Not yet claimed'],['Event inference','Aligned camera and sensor windows receive separate and fused event probabilities.','Universal detection of all road defects.'],['Impact tier','Relative Low, Medium and High priority from normalised sensor response.','Measured depth or validated engineering severity.'],['Spatial output','Each local event has a recoverable GPS record for map placement.','Final road-network accuracy after map matching.'],['Model selection','Late YOLO plus IMU has best current validation F1.','Final test performance or field reliability.']],[3.5*cm,7.1*cm,5.9*cm],y-4,s)
    p(c,'These limits identify the next evidence required: reviewed severity labels, road-network validation, threshold calibration and one final held-out test evaluation after retraining the selected components.',s['b'],2*cm,y-7,17*cm);c.showPage()
    # 8
    y=start(c,'4. GIS integration and mobile deployment design',8)
    for t in [
      'GIS is the decision layer that turns individual inferences into maintenance-relevant information. Each accepted event will carry an identifier, time, latitude, longitude, camera confidence, IMU confidence, fused probability, impact tier, session reference and map-matching metadata. Retaining the unaggregated event record enables later audit, threshold changes and comparison with field inspection without losing original evidence.',
      'The planned process first places the coordinate, then matches it to the most plausible nearby road segment while retaining distance-to-segment. Events are aggregated into 10-25 metre segments. A segment priority score combines recurrence, recency, fused confidence and provisional impact tier. This reduces duplicate alerts from repeated journeys and yields a map that emphasises road sections for review rather than isolated GPS points.']:
        y=p(c,t,s['b'],2*cm,y,17*cm)
    y=tbl(c,[['System component','Purpose','Implementation direction'],['Event store','Retain raw event evidence and outputs for audit.','Spatial database with geometry, timestamp and inference fields.'],['Map matching','Associate reports with a road segment and preserve match quality.','Road-network query with recorded match distance.'],['Segment aggregation','Convert repeated point reports into maintenance candidates.','Recurrence, recency, confidence and impact priority.'],['Analyst validation','Review clusters, source evidence and priority before action.','QGIS desktop inspection and validation.'],['Mobile map','Collect data and show segment status in field use.','React Native interface with MapLibre.']],[3.7*cm,6.7*cm,6.1*cm],y-4,s)
    y=p(c,'The mobile architecture uses React Native for the cross-platform interface, map display and user workflow. Kotlin Android modules will supply CameraX capture, high-frequency sensor access, timestamp synchronisation and TensorFlow Lite inference. This division is intentional: React Native enables rapid mobile delivery while the native layer handles timing-sensitive alignment of camera, accelerometer and gyroscope. The inference contract will pass event and probability fields to the map service, not only a binary alert.',s['b'],2*cm,y-7,17*cm)
    p(c,'The initial field interface should make uncertainty visible rather than presenting all reports as equally certain. A road segment can show its latest evidence count, its most recent event time, an aggregated confidence and its provisional impact tier. Analysts can then inspect the original event records before assigning a maintenance action. QGIS is retained for detailed desktop review because it supports controlled examination of clusters, road geometry and external reference layers.',s['b'],2*cm,y,17*cm);c.showPage()
    # 9
    y=start(c,'5. Conclusion and next implementation phase',9)
    for t in [
      'GeoFusion has moved beyond dataset inspection into a reproducible multimodal feasibility pipeline. Completed work includes sensor and annotation audit, video-time alignment, three-second RGB-IMU construction, candidate-normal selection, sensor-feature extraction, GPS recovery, compact clip preparation, session-level evaluation control, visual and inertial baselines, early and late fusion experiments, and a GIS-ready event record design.',
      'The evidence supports late YOLO plus IMU fusion as the first deployment candidate. Its validation F1 is 0.600, compared with 0.560 for early clip-plus-IMU fusion. Selection is not based on one number alone: it is supported by the advantage of retaining separate visual and inertial confidence, by the poor discriminative behaviour of the compact clip encoder, and by the fixed session-level protocol that keeps final test sessions outside the enhanced comparison.',
      'The next phase will retrain the selected visual and IMU components using training and validation sessions, perform one final evaluation on the reserved test sessions, and package the model interface for the GIS service. In parallel, the GIS schema, map matching and segment aggregation logic can be implemented, followed by a React Native and Kotlin mobile prototype. Later field collection should add reviewed severity labels and inspection outcomes so provisional impact tiers can be calibrated to a defined road-hazard scale.',
      'A controlled release sequence follows from this evidence. First, freeze the selected model and evaluate the untouched test sessions once. Second, version the inference output and store both modality probabilities with the fused score. Third, validate map matching and aggregation against a small set of road segments. Finally, place the mobile collector in supervised field trials, where uncertain and high-priority reports can be reviewed against the road surface. This sequence keeps research evaluation, GIS validation and app deployment distinct while allowing each stage to inform the next.']:
        y=p(c,t,s['b'],2*cm,y,17*cm)
    p(c,'Reference: RoadSens-4M: Road Anomaly with GIS and Weather Dataset, Scientific Data (2026), doi:10.1038/s41597-026-07072-y. Numerical results are from the project experiment under the fixed session split and validation-only selection protocol.',s['s'],2*cm,y-10,17*cm);c.save()

def merge():
    w=PdfWriter(); cover=PdfReader(str(COVER)); body=PdfReader(str(BODY)); w.add_page(cover.pages[0])
    for pg in body.pages:w.add_page(pg)
    with FINAL.open('wb') as f:w.write(f)
if __name__=='__main__':make();merge();print(FINAL)
