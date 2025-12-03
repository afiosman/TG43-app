# TG-43 Ir-192 HDR Dose Calculation Tool (v0.1.0)

A Streamlit-based educational and research tool implementing the **TG-43(U1)** dose-calculation formalism for the **Varian Ir-192 HDR brachytherapy source (VS2000)**.

---

## 🧭 Overview

This web application provides a modern, interactive implementation of the **TG-43(U1)** protocol for high-dose-rate (HDR) Ir-192 brachytherapy using the Varian VS2000 source model.

It supports:

- Point-dose evaluation for multiple dwell positions  
- 2D isodose visualization using TPS-style color maps  
- Fully interactive dwell editing (positions, dwell times, orientation angles)  
- Real-time recalculation based on TG-43 parameters  

This tool is designed for:

- Educational demonstrations  
- Dosimetry teaching labs  
- Independent verification  
- Research prototyping  
- QA method development  

⚠️ **Not a replacement for a clinically commissioned treatment-planning system (TPS).**  
See disclaimer below.

---

## ✨ Features

### ✔ TG-43(U1) Dose Engine
- Dose-rate constant  
- Radial dose function  
- 2D anisotropy function  
- Geometry factor  
- Multi-dwell support  

### ✔ Interactive UI (Streamlit-based)
- Modern VCU-style interface  
- Adjustable source activity (Ci)  
- Fully editable dwell table  
- Real-time TG-43 dose computation  

### ✔ Isodose Visualization
- Log-scaled contour map  
- Clinical colormap (blue → cyan → green → yellow → red)  
- TPS-style black isodose lines  
- Clean labeling + colorbar optimization  

### ✔ Point Dose Calculator
- Compute dose to any (x, y, z) point  
- Multi-dwell summation  

### ✔ Versioning
Current version: **v0.1.0**

![2D Isodose Example](isodose_example.png)

---
