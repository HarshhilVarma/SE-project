# Frontend Design Document — AI Image Verification System

## 1. Introduction

The proposed system is an **AI Image Verification Web Application** designed to help users determine whether an uploaded image is authentic, AI-generated, or requires further analysis.

The existing project requirements focus on binary AI-image classification, confidence scoring, visual explanations such as Grad-CAM heatmaps, and downloadable reports. The intended users include fact-checkers, journalists, content moderators, and general users.

The frontend will be intentionally **simple, minimalist, and user-focused**.

The primary verification principle is:

> **Check available provenance, watermark, and metadata information first; only perform custom visual AI detection when the available evidence is insufficient or inconclusive.**

---

## 2. Design Goals

The frontend should satisfy the following goals:

1. Simple interface — the user should immediately understand where to upload an image.
2. Minimal visual clutter — avoid unnecessary dashboards, menus, and technical information.
3. Progressive analysis — do not immediately run the custom AI detector.
4. Standards-first verification — first inspect available provenance, Content Credentials, metadata, and watermark information.
5. Transparent results — clearly distinguish:
   - Provenance information
   - Metadata/watermark findings
   - Custom AI detector findings
6. Explainable results — display confidence and heatmap information where available.
7. Clear uncertainty — absence of metadata or watermark information must not automatically be presented as proof that an image is AI-generated or authentic.
8. Responsive design — usable on desktop, tablet, and mobile.
9. Accessibility — maintain readable contrast, keyboard navigation, meaningful labels, and screen-reader-friendly controls.

The existing project specifies image upload, classification with confidence, Grad-CAM heatmaps, and downloadable reports as core user requirements.

---

## 3. Proposed Frontend Flow

```text
              ┌──────────────────┐
              │     Home Page    │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │   Upload Image   │
              └────────┬─────────┘
                       │
                       ▼
              ┌─────────────────────────┐
              │ File Validation         │
              │ JPG / PNG / WEBP        │
              └───────────┬─────────────┘
                          │
                          ▼
              ┌─────────────────────────┐
              │ Provenance & Metadata   │
              │ Check                   │
              └───────────┬─────────────┘
                          │
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
     ┌─────────────────┐      ┌────────────────────┐
     │ Useful evidence │      │ No usable evidence │
     │ found           │      │ / inconclusive     │
     └────────┬────────┘      └──────────┬─────────┘
              │                          │
              ▼                          ▼
     ┌─────────────────┐       ┌────────────────────┐
     │ Show provenance │       │ Run Custom AI      │
     │ / metadata      │       │ Detection          │
     └────────┬────────┘       └──────────┬─────────┘
              │                           │
              └─────────────┬─────────────┘
                            ▼
                 ┌─────────────────────┐
                 │ Verification Result │
                 └──────────┬──────────┘
                            │
                  ┌─────────┴─────────┐
                  ▼                   ▼
          ┌──────────────┐    ┌───────────────┐
          │ Evidence /   │    │ AI Detection  │
          │ Metadata     │    │ + Heatmap     │
          └──────────────┘    └───────────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │ Download Report    │
                  └────────────────────┘
```

The existing activity diagram already uses file validation before preprocessing and AI inference. The frontend extends this workflow by adding provenance, metadata, and watermark checking before custom AI detection.

---

## 4. Frontend Architecture

The frontend should be divided into the following UI layers:

```text
┌──────────────────────────────────────┐
│              App Shell               │
├──────────────────────────────────────┤
│ Upload / Image Selection             │
├──────────────────────────────────────┤
│ Verification Progress                │
├──────────────────────────────────────┤
│ Evidence + Detection Results         │
├──────────────────────────────────────┤
│ Report / History                     │
└──────────────────────────────────────┘
```

### Main Components

| Component | Responsibility |
|---|---|
| `Header` | Application name and minimal navigation |
| `UploadCard` | Drag/drop and file selection |
| `ImagePreview` | Display selected image |
| `FileValidation` | Display format/size validation |
| `VerificationProgress` | Show current verification stage |
| `ProvenanceCard` | Display provenance/Content Credentials |
| `MetadataCard` | Display available metadata |
| `WatermarkCard` | Display watermark findings |
| `DetectionResult` | Display AI detector result |
| `ConfidenceIndicator` | Display confidence percentage |
| `HeatmapViewer` | Display Grad-CAM/visual explanation |
| `ReportActions` | Download/share report |
| `HistoryPanel` | Display previous analyses |

The existing class design separates `User`, `MediaItem`, `DetectionEngine`, `AnalysisReport`, and `AuditLog`, which supports keeping frontend responsibilities separate from backend processing.

---

## 5. Home Page

The home page should contain very little.

### Layout

```text
                 AI VERIFY

        Verify the origin of an image

 ┌────────────────────────────────────────┐
 │                                        │
 │          Drop image here               │
 │                                        │
 │             or                         │
 │                                        │
 │        [ Choose an image ]             │
 │                                        │
 │       JPG · PNG · WEBP                 │
 │                                        │
 └────────────────────────────────────────┘

       Your image is checked for
       available provenance and metadata
       before deeper AI analysis.
```

### Design Characteristics

- White or very light background
- One primary accent color
- Rounded upload container
- Large upload icon
- Large, clear button
- No unnecessary charts
- No technical model information on the landing page

---

## 6. Image Upload Component

The upload component should support:

### Input Methods

- Drag and drop
- File picker
- Mobile file selection

### Supported Formats

```text
JPG
PNG
WEBP
```

### Initial Validation

The frontend should immediately check:

- File exists
- Supported format
- File size
- Image can be decoded
- Basic image dimensions

If validation fails:

```text
       Unable to analyze image

       Unsupported file format.

       Supported formats:
       JPG · PNG · WEBP

             [ Choose another ]
```

The project's activity diagram identifies format and size validation as an explicit stage with an error path.

---

## 7. Verification Progress Screen

After a valid image is selected, the user should see a simple progress interface.

```text
        Verifying image

        ✓ File validated
        ● Checking provenance
        ○ Reading metadata
        ○ Checking watermark information
        ○ AI image analysis
```

The important point is that **AI image analysis should not be the first operation**.

### Example States

**Stage 1**

```text
✓ Image validated
```

**Stage 2**

```text
● Checking content credentials
```

**Stage 3**

```text
● Inspecting image metadata
```

**Stage 4**

```text
● Checking available watermark information
```

**Stage 5**

```text
● Running visual AI analysis
```

Only the stages actually required should be displayed.

---

## 8. Provenance / Standards Check

This is the **first analysis layer**.

The frontend should provide a dedicated section:

```text
┌───────────────────────────────────────┐
│ Content Provenance                    │
│                                       │
│ ✓ Content Credentials found           │
│                                       │
│ Source       : Available              │
│ Created      : Available              │
│ Edited       : 2 recorded actions     │
│ AI disclosure: Available              │
│                                       │
│ [ View provenance ]                   │
└───────────────────────────────────────┘
```

Possible frontend states:

- `Verified provenance available`
- `AI creation disclosed`
- `AI modification disclosed`
- `Provenance unavailable`
- `Provenance could not be verified`

The UI should not convert the existence of provenance into an absolute real/fake judgment.

---

## 9. Metadata Check

The second layer displays conventional metadata found in the file.

Example:

```text
┌───────────────────────────────────────┐
│ Image Metadata                        │
│                                       │
│ Camera          Sony                  │
│ Software        Photoshop             │
│ Date Created    14 Sep 2026           │
│ Dimensions      1920 × 1080            │
│ Color Space     sRGB                  │
│                                       │
│ [ Show all metadata ]                 │
└───────────────────────────────────────┘
```

Metadata should be presented as **evidence**, not automatically as proof.

---

## 10. Watermark Check

A separate but compact section should be provided.

```text
┌───────────────────────────────────────┐
│ Watermark / Embedded Signals          │
│                                       │
│ Status: No verified watermark found  │
│                                       │
│ This does not determine whether the  │
│ image is AI-generated.                │
└───────────────────────────────────────┘
```

### Possible States

#### Found

```text
✓ Signal detected

A recognized watermark/provenance signal
was detected.

[ View details ]
```

#### Not Found

```text
— No recognized signal found

No supported watermark signal was detected.
This does not indicate that the image is real
or AI-generated.
```

#### Unable to Determine

```text
? Unable to determine

The available information was insufficient
for verification.
```

---

## 11. Decision Point

The frontend should determine which result interface to show based on the verification response.

### Case A — Useful provenance information

```text
Image
  ↓
Provenance found
  ↓
Display provenance
  ↓
Optional deeper analysis
```

### Case B — No usable provenance

```text
Image
  ↓
No usable provenance
  ↓
Metadata / watermark inconclusive
  ↓
Custom AI detector
```

This allows the application to follow the requested **standards-first** architecture without removing the existing AI detection functionality.

---

## 12. Custom AI Detection Screen

When deeper analysis is required, the interface should remain simple.

```text
        Analyzing image...

        ███████████░░░  78%

        Examining visual patterns
```

Avoid displaying technical model names such as:

```text
ResNet-50
EfficientNet
CNN
PyTorch
```

on the primary interface.

Those details can be placed inside an optional **Technical Details** section.

---

## 13. Main Result Screen

The result page should be the most important screen.

### Example

```text
┌─────────────────────────────────────────────┐
│                                             │
│              Verification Result            │
│                                             │
│                 AI-GENERATED                │
│                                             │
│                    92%                      │
│                  confidence                 │
│                                             │
│       Custom visual analysis indicates      │
│       patterns associated with AI           │
│       generated imagery.                    │
│                                             │
└─────────────────────────────────────────────┘


        [ Original ]       [ Heatmap ]


       ┌──────────────────────────┐
       │                          │
       │        IMAGE             │
       │                          │
       └──────────────────────────┘


       Provenance
       No verified provenance found

       Metadata
       3 metadata fields detected

       Watermark
       No supported signal detected


       [ Download Report ]
```

The existing requirements specify a clear classification result with a confidence percentage and a visual heatmap.

---

## 14. Result Classification

Use a small number of result states.

### AI-Generated

```text
AI-GENERATED
92% confidence
```

### Authentic

```text
AUTHENTIC
78% confidence
```

### Inconclusive

```text
INCONCLUSIVE

The available evidence does not provide
sufficient confidence for classification.
```

The existing sequence diagram uses an alternative path based on the confidence threshold, so the frontend should be capable of displaying different classification states.

---

## 15. Heatmap Viewer

The project uses Grad-CAM for visual explanation.

The frontend should provide a simple toggle:

```text
[ Original ] [ Heatmap ] [ Overlay ]
```

Example:

```text
┌─────────────────────────────────┐
│                                 │
│         Image + Heatmap         │
│                                 │
│     suspicious regions shown    │
│     using visual overlay        │
│                                 │
└─────────────────────────────────┘

Heatmap intensity indicates regions
used by the model during analysis.
```

The UI should describe highlighted areas as:

> **Regions that contributed to the model's prediction**

rather than automatically labeling them as definitely fake.

---

## 16. Evidence Summary

The result page should have a compact evidence summary.

```text
Evidence Summary

✓ Provenance        Available
✓ Metadata          Available
— Watermark         Not detected
● AI Analysis       92% AI-generated
```

Clicking each item opens more information.

This creates a clean **progressive disclosure** design.

---

## 17. Detailed Analysis Drawer

Clicking `View details` opens a side panel.

```text
┌──────────────────────────────────┐
│ Analysis Details             ×   │
├──────────────────────────────────┤
│ Provenance                       │
│                                  │
│ Content Credentials              │
│ Status: Verified                 │
│                                  │
│ Metadata                         │
│                                  │
│ Software: XXXXX                  │
│ Created: XXXXX                   │
│ Dimensions: XXXXX                │
│                                  │
│ AI Analysis                      │
│                                  │
│ Confidence: 92%                 │
│ Model: [Technical Details]      │
│                                  │
└──────────────────────────────────┘
```

This keeps the main interface minimalist while allowing advanced users to inspect more information.

---

## 18. Downloadable Report

The existing project requirements include downloading a basic PDF summary report.

The frontend should provide:

```text
[ Download Report ]
```

The report should summarize:

- Image preview
- File information
- Verification timestamp
- Provenance result
- Metadata result
- Watermark result
- AI detection result
- Confidence score
- Heatmap
- Important limitations

Example:

```text
AI IMAGE VERIFICATION REPORT

Classification
AI-Generated

Confidence
92%

Provenance
No verified provenance found

Metadata
Available

Watermark
Not detected

Visual Analysis
See attached heatmap
```

---

## 19. History Page

History should be secondary rather than part of the main landing page.

```text
History

┌────────────┬──────────────┬─────────────┐
│ Image      │ Result       │ Date        │
├────────────┼──────────────┼─────────────┤
│ image1.jpg │ AI-Generated │ 16 Sep 2026 │
│ image2.png │ Inconclusive │ 15 Sep 2026 │
│ image3.jpg │ Authentic    │ 14 Sep 2026 │
└────────────┴──────────────┴─────────────┘
```

Selecting an item opens its report.

---

## 20. Navigation

Keep navigation extremely small.

```text
AI VERIFY

Verify       History
```

Optional:

```text
AI VERIFY

Verify   History   About
```

There should be no complex sidebar in the initial frontend.

Admin and security interfaces are outside the scope of this frontend document.

---

## 21. Visual Design System

### Colors

Use a restrained palette:

| Element | Suggested Style |
|---|---|
| Background | Off-white / very light gray |
| Primary | Dark neutral or single accent |
| Text | Near-black |
| Secondary text | Gray |
| Cards | White |
| Borders | Light gray |
| Success | Subtle green |
| Warning | Subtle amber |
| Error | Subtle red |

Avoid a highly colorful "AI dashboard" aesthetic.

### Typography

Use a clean sans-serif font.

```text
Page title       32 px
Section title    20–24 px
Body             15–16 px
Supporting text  13–14 px
```

Use generous whitespace rather than decorative elements.

---

## 22. Component Design

### Primary Button

```text
┌────────────────────┐
│   Choose Image     │
└────────────────────┘
```

### Secondary Button

```text
[ View Details ]
```

### Status Indicator

```text
● Checking
✓ Verified
— Not found
! Attention
```

### Cards

Cards should have:

- 12–16 px border radius
- Thin border
- Minimal shadow
- Consistent padding
- Clear heading

---

## 23. Responsive Layout

### Desktop

```text
┌───────────────────────────────────────────────┐
│ Header                                        │
├───────────────────────┬───────────────────────┤
│                       │                       │
│ Image                 │ Verification         │
│                       │ Summary               │
│                       │                       │
└───────────────────────┴───────────────────────┘
```

### Mobile

```text
┌───────────────────────┐
│ Header                │
├───────────────────────┤
│ Image                 │
├───────────────────────┤
│ Result                │
├───────────────────────┤
│ Provenance            │
├───────────────────────┤
│ Metadata              │
├───────────────────────┤
│ AI Analysis           │
└───────────────────────┘
```

---

## 24. Frontend States

The frontend should explicitly support these states:

| State | UI |
|---|---|
| Empty | Upload image |
| Uploading | Upload progress |
| Invalid file | Validation error |
| Checking provenance | Loading state |
| Reading metadata | Loading state |
| Checking watermark | Loading state |
| Provenance found | Evidence card |
| No provenance | Neutral information |
| Running AI detector | Analysis progress |
| Result available | Result dashboard |
| Inconclusive | Neutral result |
| Backend error | Retry message |

The project documentation emphasizes dedicated error paths and validation gates for invalid input.

---

## 25. Important UX Rule: Evidence ≠ Verdict

This should be a central design principle.

The interface should visually separate:

```text
PROVENANCE
     ↓
METADATA
     ↓
WATERMARK
     ↓
VISUAL AI ANALYSIS
     ↓
OVERALL REPORT
```

### Good

```text
Content Credentials: Verified
AI disclosure: Present
Visual detector: 92% AI-generated
```

### Avoid

```text
C2PA found → REAL
```

or:

```text
No metadata → FAKE
```

The frontend should clearly distinguish technical evidence from the detector's classification.

---

## 26. Accessibility

The frontend should follow accessible design principles:

- Keyboard-accessible upload button
- Visible focus indicators
- Proper labels for controls
- Alt text for image previews
- Do not rely only on color to communicate status
- Readable text sizes
- Sufficient contrast
- Screen-reader-friendly progress messages
- Avoid excessive animation

---

## 27. Frontend Scope

### In Scope

- Landing page
- Image upload
- Drag-and-drop
- Image preview
- File validation UI
- Verification progress
- Provenance/Content Credentials UI
- Metadata UI
- Watermark status UI
- Custom AI detection result UI
- Confidence score
- Heatmap viewer
- Evidence summary
- Detailed analysis panel
- PDF report button
- Basic history
- Responsive design
- Error handling

### Out of Scope

- AI model training
- Dataset preparation
- Model architecture
- Model inference implementation
- Metadata extraction implementation
- C2PA verification implementation
- Watermark extraction algorithm
- Database implementation
- Authentication backend
- Admin backend
- Security subsystem
- Deepfake video detection

---

## 28. Recommended Frontend Technology

The existing project identifies **Streamlit or React** as feasible frontend options.

For the final product, the frontend can be structured as a **React-based web application** because the proposed interface requires:

- Multiple UI states
- Progressive verification
- Expandable evidence panels
- Image/heatmap comparison
- Responsive layouts
- Reusable components
- Separation between frontend and backend APIs

The backend can remain responsible for the actual verification and AI processing.

```text
Frontend
   │
   ├── Upload
   ├── Display
   ├── State management
   └── User interaction
          │
          ▼
       API Layer
          │
          ├── Provenance checker
          ├── Metadata checker
          ├── Watermark checker
          └── AI detection engine
```

---

## 29. Final Frontend Concept

The complete user experience should feel like this:

```text
                AI VERIFY

       "Check an image before trusting it."

                     │
                     ▼

             ┌───────────────┐
             │ Upload Image  │
             └───────┬───────┘
                     │
                     ▼
          ┌──────────────────────┐
          │ Validate File        │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │ Provenance / C2PA    │
          │ Metadata / Watermark │
          └──────────┬───────────┘
                     │
              Evidence found?
                 /         \
               Yes          No
                │            │
                │            ▼
                │     Custom AI Analysis
                │            │
                └──────┬─────┘
                       ▼
              ┌─────────────────┐
              │ Verification    │
              │ Result          │
              └────────┬────────┘
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
        Evidence             Heatmap
        Details              Analysis
             │                   │
             └─────────┬─────────┘
                       ▼
                Download Report
```

## 30. Design Principle

> **Minimal interface, maximum clarity.**

The frontend should not look like a complicated forensic dashboard. A general user should be able to upload an image and understand the result quickly, while a journalist or fact-checker can expand the interface to inspect provenance, metadata, watermark information, confidence, and visual evidence.

The design builds on the project's existing requirements, activity flow, use cases, and class structure while limiting this document to the **frontend layer**.
