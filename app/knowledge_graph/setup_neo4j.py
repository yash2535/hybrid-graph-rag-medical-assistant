from neo4j import GraphDatabase
import os
from datetime import datetime, timedelta

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

print("=" * 60)
print("SETTING UP NEO4J MEDICAL KNOWLEDGE GRAPH - MULTI-PATIENT")
print("=" * 60)

with driver.session() as session:
    # Clear existing data
    print("\n[1/6] Clearing existing data...")
    session.run("MATCH (n) DETACH DELETE n")
    print("✓ Database cleared")
    
    # ============================================================
    # PATIENT 1: John Doe - Type 2 Diabetes + Hypertension
    # ============================================================
    print("\n[2/6] Creating Patient 1: John Doe (Type 2 Diabetes + Hypertension)...")
    
    session.run("""
        CREATE (p:Patient {
          id: "user_1",
          name: "John Doe",
          age: 52,
          gender: "Male",
          bloodType: "O+",
          created_at: datetime()
        })
        
        CREATE (d1:Disease {
          id: "d1_p1",
          name: "Type 2 Diabetes",
          category: "Endocrine",
          severity: "moderate",
          status: "active",
          diagnosisDate: date("2018-06-20"),
          icd10: "E11.9"
        })
        
        CREATE (d2:Disease {
          id: "d2_p1",
          name: "Hypertension",
          category: "Cardiovascular",
          severity: "moderate",
          status: "active",
          diagnosisDate: date("2019-03-15"),
          icd10: "I10"
        })
        
        CREATE (m1:Medication {
          id: "m1_p1",
          name: "Metformin",
          dosage: "1000mg",
          frequency: "twice daily",
          purpose: "Blood sugar control",
          generic: true,
          atcCode: "A10BA02"
        })
        
        CREATE (m2:Medication {
          id: "m2_p1",
          name: "Lisinopril",
          dosage: "10mg",
          frequency: "once daily",
          purpose: "Blood pressure control",
          generic: true,
          atcCode: "C09AA03"
        })
        
        CREATE (m3:Medication {
          id: "m3_p1",
          name: "Amlodipine",
          dosage: "5mg",
          frequency: "once daily",
          purpose: "Blood pressure control",
          generic: true,
          atcCode: "C08CA01"
        })
        
        CREATE (l1:LabTest {
          id: "l1_p1",
          name: "HbA1c",
          result: 7.2,
          unit: "%",
          normalRange: "< 5.7%",
          status: "slightly elevated",
          testDate: date("2025-01-10"),
          testCode: "2345-7"
        })
        
        CREATE (l2:LabTest {
          id: "l2_p1",
          name: "Fasting Glucose",
          result: 135,
          unit: "mg/dL",
          normalRange: "70-100 mg/dL",
          status: "elevated",
          testDate: date("2025-01-10"),
          testCode: "1558-6"
        })
        
        CREATE (l3:LabTest {
          id: "l3_p1",
          name: "Blood Pressure",
          result: "142/88",
          unit: "mmHg",
          normalRange: "< 120/80 mmHg",
          status: "stage 1 hypertension",
          testDate: date("2025-01-10"),
          testCode: "55284-4"
        })
        
        CREATE (l4:LabTest {
          id: "l4_p1",
          name: "Total Cholesterol",
          result: 210,
          unit: "mg/dL",
          normalRange: "< 200 mg/dL",
          status: "borderline high",
          testDate: date("2025-01-10"),
          testCode: "2093-3"
        })
    """)
    
    # Link Patient 1 data
    session.run("""
        MATCH (p:Patient {id: "user_1"}), 
              (d1:Disease {id: "d1_p1"}), (d2:Disease {id: "d2_p1"}),
              (m1:Medication {id: "m1_p1"}), (m2:Medication {id: "m2_p1"}), (m3:Medication {id: "m3_p1"}),
              (l1:LabTest {id: "l1_p1"}), (l2:LabTest {id: "l2_p1"}), (l3:LabTest {id: "l3_p1"}), (l4:LabTest {id: "l4_p1"})
        CREATE (p)-[:HAS_DISEASE]->(d1)
        CREATE (p)-[:HAS_DISEASE]->(d2)
        CREATE (d1)-[:TREATED_BY]->(m1)
        CREATE (d2)-[:TREATED_BY]->(m2)
        CREATE (d2)-[:TREATED_BY]->(m3)
        CREATE (p)-[:PRESCRIBED]->(m1)
        CREATE (p)-[:PRESCRIBED]->(m2)
        CREATE (p)-[:PRESCRIBED]->(m3)
        CREATE (d1)-[:HAS_LAB_RESULT]->(l1)
        CREATE (d1)-[:HAS_LAB_RESULT]->(l2)
        CREATE (d2)-[:HAS_LAB_RESULT]->(l3)
        CREATE (d2)-[:HAS_LAB_RESULT]->(l4)
    """)
    
    # Create wearable metrics and readings for Patient 1
    session.run("""
        CREATE (wm1:WearableMetric {
          id: "wm1_p1",
          type: "blood_glucose",
          name: "Blood Glucose",
          unit: "mg/dL",
          normalRange: "70-100 (fasting)"
        })
        CREATE (wm2:WearableMetric {
          id: "wm2_p1",
          type: "blood_pressure",
          name: "Blood Pressure",
          unit: "mmHg",
          normalRange: "< 120/80"
        })
        CREATE (wm3:WearableMetric {
          id: "wm3_p1",
          type: "heart_rate",
          name: "Heart Rate",
          unit: "bpm",
          normalRange: "60-100"
        })
        CREATE (wm4:WearableMetric {
          id: "wm4_p1",
          type: "steps",
          name: "Daily Steps",
          unit: "steps",
          normalRange: "> 8000"
        })
    """)
    
    session.run("""
        MATCH (p:Patient {id: "user_1"}), 
              (wm1:WearableMetric {id: "wm1_p1"}), (wm2:WearableMetric {id: "wm2_p1"}),
              (wm3:WearableMetric {id: "wm3_p1"}), (wm4:WearableMetric {id: "wm4_p1"})
        CREATE (p)-[:HAS_METRIC]->(wm1)
        CREATE (p)-[:HAS_METRIC]->(wm2)
        CREATE (p)-[:HAS_METRIC]->(wm3)
        CREATE (p)-[:HAS_METRIC]->(wm4)
    """)
    
    # Readings for Patient 1
    session.run("""
        CREATE (r1:Reading {id: "r_bg_1_p1", value: 156, timestamp: datetime("2026-02-08T08:00:00Z")})
        CREATE (r2:Reading {id: "r_bg_2_p1", value: 142, timestamp: datetime("2026-02-09T08:00:00Z")})
        CREATE (r3:Reading {id: "r_bp_1_p1", value: "138/88", timestamp: datetime("2026-02-08T08:00:00Z")})
        CREATE (r4:Reading {id: "r_bp_2_p1", value: "142/90", timestamp: datetime("2026-02-09T08:00:00Z")})
        CREATE (r5:Reading {id: "r_hr_1_p1", value: 72, timestamp: datetime("2026-02-08T08:00:00Z")})
        CREATE (r6:Reading {id: "r_hr_2_p1", value: 75, timestamp: datetime("2026-02-09T08:00:00Z")})
        CREATE (r7:Reading {id: "r_steps_1_p1", value: 8234, timestamp: datetime("2026-02-08T23:59:00Z")})
        CREATE (r8:Reading {id: "r_steps_2_p1", value: 6500, timestamp: datetime("2026-02-09T23:59:00Z")})
    """)
    
    session.run("""
        MATCH (wm:WearableMetric {id: "wm1_p1"}), (r:Reading) 
        WHERE r.id IN ["r_bg_1_p1", "r_bg_2_p1"]
        CREATE (wm)-[:RECORDED_AS]->(r)
    """)
    session.run("""
        MATCH (wm:WearableMetric {id: "wm2_p1"}), (r:Reading) 
        WHERE r.id IN ["r_bp_1_p1", "r_bp_2_p1"]
        CREATE (wm)-[:RECORDED_AS]->(r)
    """)
    session.run("""
        MATCH (wm:WearableMetric {id: "wm3_p1"}), (r:Reading) 
        WHERE r.id IN ["r_hr_1_p1", "r_hr_2_p1"]
        CREATE (wm)-[:RECORDED_AS]->(r)
    """)
    session.run("""
        MATCH (wm:WearableMetric {id: "wm4_p1"}), (r:Reading) 
        WHERE r.id IN ["r_steps_1_p1", "r_steps_2_p1"]
        CREATE (wm)-[:RECORDED_AS]->(r)
    """)
    
    print("✓ Patient 1 complete with all nodes")
    
    # ============================================================
    # PATIENT 2: Sarah Smith - Heart Disease
    # ============================================================
    print("\n[3/6] Creating Patient 2: Sarah Smith (Heart Disease)...")
    
    session.run("""
        CREATE (p:Patient {
          id: "user_2",
          name: "Sarah Smith",
          age: 58,
          gender: "Female",
          bloodType: "A+",
          created_at: datetime()
        })
        
        CREATE (d1:Disease {
          id: "d1_p2",
          name: "Heart Disease",
          category: "Cardiovascular",
          severity: "moderate",
          status: "active",
          diagnosisDate: date("2020-01-15"),
          icd10: "I25.10"
        })
        
        CREATE (m1:Medication {
          id: "m1_p2",
          name: "Atorvastatin",
          dosage: "40mg",
          frequency: "once daily",
          purpose: "Cholesterol management",
          generic: true,
          atcCode: "C10AA05"
        })
        
        CREATE (m2:Medication {
          id: "m2_p2",
          name: "Aspirin",
          dosage: "81mg",
          frequency: "once daily",
          purpose: "Blood thinner",
          generic: true,
          atcCode: "B01AC06"
        })
        
        CREATE (m3:Medication {
          id: "m3_p2",
          name: "Metoprolol",
          dosage: "50mg",
          frequency: "twice daily",
          purpose: "Heart rate control",
          generic: true,
          atcCode: "C07AB02"
        })
        
        CREATE (l1:LabTest {
          id: "l1_p2",
          name: "Troponin",
          result: 0.02,
          unit: "ng/mL",
          normalRange: "< 0.04 ng/mL",
          status: "normal",
          testDate: date("2025-01-08"),
          testCode: "6598-7"
        })
        
        CREATE (l2:LabTest {
          id: "l2_p2",
          name: "BNP",
          result: 120,
          unit: "pg/mL",
          normalRange: "< 100 pg/mL",
          status: "slightly elevated",
          testDate: date("2025-01-08"),
          testCode: "35637-4"
        })
        
        CREATE (l3:LabTest {
          id: "l3_p2",
          name: "LDL Cholesterol",
          result: 145,
          unit: "mg/dL",
          normalRange: "< 100 mg/dL",
          status: "elevated",
          testDate: date("2025-01-08"),
          testCode: "18262-6"
        })
        
        CREATE (l4:LabTest {
          id: "l4_p2",
          name: "HDL Cholesterol",
          result: 42,
          unit: "mg/dL",
          normalRange: "> 40 mg/dL",
          status: "borderline low",
          testDate: date("2025-01-08"),
          testCode: "2085-9"
        })
    """)
    
    session.run("""
        MATCH (p:Patient {id: "user_2"}), 
              (d1:Disease {id: "d1_p2"}),
              (m1:Medication {id: "m1_p2"}), (m2:Medication {id: "m2_p2"}), (m3:Medication {id: "m3_p2"}),
              (l1:LabTest {id: "l1_p2"}), (l2:LabTest {id: "l2_p2"}), (l3:LabTest {id: "l3_p2"}), (l4:LabTest {id: "l4_p2"})
        CREATE (p)-[:HAS_DISEASE]->(d1)
        CREATE (d1)-[:TREATED_BY]->(m1)
        CREATE (d1)-[:TREATED_BY]->(m2)
        CREATE (d1)-[:TREATED_BY]->(m3)
        CREATE (p)-[:PRESCRIBED]->(m1)
        CREATE (p)-[:PRESCRIBED]->(m2)
        CREATE (p)-[:PRESCRIBED]->(m3)
        CREATE (d1)-[:HAS_LAB_RESULT]->(l1)
        CREATE (d1)-[:HAS_LAB_RESULT]->(l2)
        CREATE (d1)-[:HAS_LAB_RESULT]->(l3)
        CREATE (d1)-[:HAS_LAB_RESULT]->(l4)
    """)
    
    # Wearables for Patient 2
    session.run("""
        CREATE (wm1:WearableMetric {
          id: "wm1_p2",
          type: "heart_rate",
          name: "Heart Rate",
          unit: "bpm",
          normalRange: "60-100"
        })
        CREATE (wm2:WearableMetric {
          id: "wm2_p2",
          type: "blood_pressure",
          name: "Blood Pressure",
          unit: "mmHg",
          normalRange: "< 120/80"
        })
        CREATE (wm3:WearableMetric {
          id: "wm3_p2",
          type: "ecg",
          name: "ECG",
          unit: "rhythm",
          normalRange: "normal sinus rhythm"
        })
        CREATE (wm4:WearableMetric {
          id: "wm4_p2",
          type: "steps",
          name: "Daily Steps",
          unit: "steps",
          normalRange: "> 6000"
        })
    """)
    
    session.run("""
        MATCH (p:Patient {id: "user_2"}), 
              (wm1:WearableMetric {id: "wm1_p2"}), (wm2:WearableMetric {id: "wm2_p2"}),
              (wm3:WearableMetric {id: "wm3_p2"}), (wm4:WearableMetric {id: "wm4_p2"})
        CREATE (p)-[:HAS_METRIC]->(wm1)
        CREATE (p)-[:HAS_METRIC]->(wm2)
        CREATE (p)-[:HAS_METRIC]->(wm3)
        CREATE (p)-[:HAS_METRIC]->(wm4)
    """)
    
    session.run("""
        CREATE (r1:Reading {id: "r_hr_1_p2", value: 68, timestamp: datetime("2026-02-08T08:00:00Z")})
        CREATE (r2:Reading {id: "r_hr_2_p2", value: 72, timestamp: datetime("2026-02-09T08:00:00Z")})
        CREATE (r3:Reading {id: "r_bp_1_p2", value: "125/82", timestamp: datetime("2026-02-08T08:00:00Z")})
        CREATE (r4:Reading {id: "r_bp_2_p2", value: "128/85", timestamp: datetime("2026-02-09T08:00:00Z")})
        CREATE (r5:Reading {id: "r_ecg_1_p2", value: "NSR", timestamp: datetime("2026-02-08T08:00:00Z")})
        CREATE (r6:Reading {id: "r_ecg_2_p2", value: "NSR", timestamp: datetime("2026-02-09T08:00:00Z")})
        CREATE (r7:Reading {id: "r_steps_1_p2", value: 5234, timestamp: datetime("2026-02-08T23:59:00Z")})
        CREATE (r8:Reading {id: "r_steps_2_p2", value: 6100, timestamp: datetime("2026-02-09T23:59:00Z")})
    """)
    
    session.run("""MATCH (wm:WearableMetric {id: "wm1_p2"}), (r:Reading) WHERE r.id IN ["r_hr_1_p2", "r_hr_2_p2"] CREATE (wm)-[:RECORDED_AS]->(r)""")
    session.run("""MATCH (wm:WearableMetric {id: "wm2_p2"}), (r:Reading) WHERE r.id IN ["r_bp_1_p2", "r_bp_2_p2"] CREATE (wm)-[:RECORDED_AS]->(r)""")
    session.run("""MATCH (wm:WearableMetric {id: "wm3_p2"}), (r:Reading) WHERE r.id IN ["r_ecg_1_p2", "r_ecg_2_p2"] CREATE (wm)-[:RECORDED_AS]->(r)""")
    session.run("""MATCH (wm:WearableMetric {id: "wm4_p2"}), (r:Reading) WHERE r.id IN ["r_steps_1_p2", "r_steps_2_p2"] CREATE (wm)-[:RECORDED_AS]->(r)""")
    
    print("✓ Patient 2 complete with all nodes")
    
    # ============================================================
    # PATIENT 3: Michael Johnson - Asthma
    # ============================================================
    print("\n[4/6] Creating Patient 3: Michael Johnson (Asthma)...")
    
    session.run("""
        CREATE (p:Patient {
          id: "user_3",
          name: "Michael Johnson",
          age: 34,
          gender: "Male",
          bloodType: "B+",
          created_at: datetime()
        })
        
        CREATE (d1:Disease {
          id: "d1_p3",
          name: "Asthma",
          category: "Respiratory",
          severity: "moderate",
          status: "active",
          diagnosisDate: date("2015-09-12"),
          icd10: "J45.20"
        })
        
        CREATE (m1:Medication {
          id: "m1_p3",
          name: "Albuterol Inhaler",
          dosage: "90mcg",
          frequency: "as needed",
          purpose: "Quick relief bronchodilator",
          generic: true,
          atcCode: "R03AC02"
        })
        
        CREATE (m2:Medication {
          id: "m2_p3",
          name: "Fluticasone Inhaler",
          dosage: "220mcg",
          frequency: "twice daily",
          purpose: "Long-term asthma control",
          generic: false,
          atcCode: "R03BA05"
        })
        
        CREATE (m3:Medication {
          id: "m3_p3",
          name: "Montelukast",
          dosage: "10mg",
          frequency: "once daily",
          purpose: "Asthma control",
          generic: true,
          atcCode: "R03DC03"
        })
        
        CREATE (l1:LabTest {
          id: "l1_p3",
          name: "Spirometry FEV1",
          result: 78,
          unit: "% predicted",
          normalRange: "> 80%",
          status: "slightly reduced",
          testDate: date("2025-01-05"),
          testCode: "20150-9"
        })
        
        CREATE (l2:LabTest {
          id: "l2_p3",
          name: "Peak Flow",
          result: 420,
          unit: "L/min",
          normalRange: "450-550 L/min",
          status: "reduced",
          testDate: date("2025-01-05"),
          testCode: "19935-6"
        })
        
        CREATE (l3:LabTest {
          id: "l3_p3",
          name: "IgE Total",
          result: 245,
          unit: "IU/mL",
          normalRange: "< 100 IU/mL",
          status: "elevated",
          testDate: date("2025-01-05"),
          testCode: "19113-0"
        })
        
        CREATE (l4:LabTest {
          id: "l4_p3",
          name: "Eosinophil Count",
          result: 520,
          unit: "cells/μL",
          normalRange: "< 500 cells/μL",
          status: "slightly elevated",
          testDate: date("2025-01-05"),
          testCode: "713-8"
        })
    """)
    
    session.run("""
        MATCH (p:Patient {id: "user_3"}), 
              (d1:Disease {id: "d1_p3"}),
              (m1:Medication {id: "m1_p3"}), (m2:Medication {id: "m2_p3"}), (m3:Medication {id: "m3_p3"}),
              (l1:LabTest {id: "l1_p3"}), (l2:LabTest {id: "l2_p3"}), (l3:LabTest {id: "l3_p3"}), (l4:LabTest {id: "l4_p3"})
        CREATE (p)-[:HAS_DISEASE]->(d1)
        CREATE (d1)-[:TREATED_BY]->(m1)
        CREATE (d1)-[:TREATED_BY]->(m2)
        CREATE (d1)-[:TREATED_BY]->(m3)
        CREATE (p)-[:PRESCRIBED]->(m1)
        CREATE (p)-[:PRESCRIBED]->(m2)
        CREATE (p)-[:PRESCRIBED]->(m3)
        CREATE (d1)-[:HAS_LAB_RESULT]->(l1)
        CREATE (d1)-[:HAS_LAB_RESULT]->(l2)
        CREATE (d1)-[:HAS_LAB_RESULT]->(l3)
        CREATE (d1)-[:HAS_LAB_RESULT]->(l4)
    """)
    
    # Wearables for Patient 3
    session.run("""
        CREATE (wm1:WearableMetric {
          id: "wm1_p3",
          type: "peak_flow",
          name: "Peak Flow",
          unit: "L/min",
          normalRange: "450-550"
        })
        CREATE (wm2:WearableMetric {
          id: "wm2_p3",
          type: "respiratory_rate",
          name: "Respiratory Rate",
          unit: "breaths/min",
          normalRange: "12-20"
        })
        CREATE (wm3:WearableMetric {
          id: "wm3_p3",
          type: "spo2",
          name: "Blood Oxygen",
          unit: "%",
          normalRange: "> 95%"
        })
        CREATE (wm4:WearableMetric {
          id: "wm4_p3",
          type: "steps",
          name: "Daily Steps",
          unit: "steps",
          normalRange: "> 10000"
        })
    """)
    
    session.run("""
        MATCH (p:Patient {id: "user_3"}), 
              (wm1:WearableMetric {id: "wm1_p3"}), (wm2:WearableMetric {id: "wm2_p3"}),
              (wm3:WearableMetric {id: "wm3_p3"}), (wm4:WearableMetric {id: "wm4_p3"})
        CREATE (p)-[:HAS_METRIC]->(wm1)
        CREATE (p)-[:HAS_METRIC]->(wm2)
        CREATE (p)-[:HAS_METRIC]->(wm3)
        CREATE (p)-[:HAS_METRIC]->(wm4)
    """)
    
    session.run("""
        CREATE (r1:Reading {id: "r_pf_1_p3", value: 420, timestamp: datetime("2026-02-08T08:00:00Z")})
        CREATE (r2:Reading {id: "r_pf_2_p3", value: 435, timestamp: datetime("2026-02-09T08:00:00Z")})
        CREATE (r3:Reading {id: "r_rr_1_p3", value: 16, timestamp: datetime("2026-02-08T08:00:00Z")})
        CREATE (r4:Reading {id: "r_rr_2_p3", value: 15, timestamp: datetime("2026-02-09T08:00:00Z")})
        CREATE (r5:Reading {id: "r_spo2_1_p3", value: 97, timestamp: datetime("2026-02-08T08:00:00Z")})
        CREATE (r6:Reading {id: "r_spo2_2_p3", value: 98, timestamp: datetime("2026-02-09T08:00:00Z")})
        CREATE (r7:Reading {id: "r_steps_1_p3", value: 11234, timestamp: datetime("2026-02-08T23:59:00Z")})
        CREATE (r8:Reading {id: "r_steps_2_p3", value: 9800, timestamp: datetime("2026-02-09T23:59:00Z")})
    """)
    
    session.run("""MATCH (wm:WearableMetric {id: "wm1_p3"}), (r:Reading) WHERE r.id IN ["r_pf_1_p3", "r_pf_2_p3"] CREATE (wm)-[:RECORDED_AS]->(r)""")
    session.run("""MATCH (wm:WearableMetric {id: "wm2_p3"}), (r:Reading) WHERE r.id IN ["r_rr_1_p3", "r_rr_2_p3"] CREATE (wm)-[:RECORDED_AS]->(r)""")
    session.run("""MATCH (wm:WearableMetric {id: "wm3_p3"}), (r:Reading) WHERE r.id IN ["r_spo2_1_p3", "r_spo2_2_p3"] CREATE (wm)-[:RECORDED_AS]->(r)""")
    session.run("""MATCH (wm:WearableMetric {id: "wm4_p3"}), (r:Reading) WHERE r.id IN ["r_steps_1_p3", "r_steps_2_p3"] CREATE (wm)-[:RECORDED_AS]->(r)""")
    
    print("✓ Patient 3 complete with all nodes")
    
    # ============================================================
    # PATIENT 4: Emily Davis - Chronic Kidney Disease
    # ============================================================
    print("\n[5/6] Creating Patient 4: Emily Davis (Chronic Kidney Disease)...")
    
    session.run("""
        CREATE (p:Patient {
          id: "user_4",
          name: "Emily Davis",
          age: 64,
          gender: "Female",
          bloodType: "AB+",
          created_at: datetime()
        })
        
        CREATE (d1:Disease {
          id: "d1_p4",
          name: "Chronic Kidney Disease",
          category: "Renal",
          severity: "moderate",
          status: "active",
          diagnosisDate: date("2021-11-03"),
          icd10: "N18.3"
        })
        
        CREATE (m1:Medication {
          id: "m1_p4",
          name: "Losartan",
          dosage: "50mg",
          frequency: "once daily",
          purpose: "Blood pressure and kidney protection",
          generic: true,
          atcCode: "C09CA01"
        })
        
        CREATE (m2:Medication {
          id: "m2_p4",
          name: "Furosemide",
          dosage: "40mg",
          frequency: "once daily",
          purpose: "Fluid management",
          generic: true,
          atcCode: "C03CA01"
        })
        
        CREATE (m3:Medication {
          id: "m3_p4",
          name: "Calcium Carbonate",
          dosage: "500mg",
          frequency: "three times daily",
          purpose: "Phosphate binder",
          generic: true,
          atcCode: "A12AA04"
        })
        
        CREATE (m4:Medication {
          id: "m4_p4",
          name: "Erythropoietin",
          dosage: "4000 units",
          frequency: "weekly injection",
          purpose: "Anemia management",
          generic: false,
          atcCode: "B03XA01"
        })
        
        CREATE (l1:LabTest {
          id: "l1_p4",
          name: "Creatinine",
          result: 2.8,
          unit: "mg/dL",
          normalRange: "0.7-1.3 mg/dL",
          status: "elevated",
          testDate: date("2025-01-12"),
          testCode: "2160-0"
        })
        
        CREATE (l2:LabTest {
          id: "l2_p4",
          name: "eGFR",
          result: 42,
          unit: "mL/min/1.73m²",
          normalRange: "> 90 mL/min/1.73m²",
          status: "stage 3 CKD",
          testDate: date("2025-01-12"),
          testCode: "33914-3"
        })
        
        CREATE (l3:LabTest {
          id: "l3_p4",
          name: "Potassium",
          result: 5.2,
          unit: "mEq/L",
          normalRange: "3.5-5.0 mEq/L",
          status: "slightly elevated",
          testDate: date("2025-01-12"),
          testCode: "2823-3"
        })
        
        CREATE (l4:LabTest {
          id: "l4_p4",
          name: "Phosphorus",
          result: 5.8,
          unit: "mg/dL",
          normalRange: "2.5-4.5 mg/dL",
          status: "elevated",
          testDate: date("2025-01-12"),
          testCode: "2777-1"
        })
        
        CREATE (l5:LabTest {
          id: "l5_p4",
          name: "Hemoglobin",
          result: 10.2,
          unit: "g/dL",
          normalRange: "12-16 g/dL",
          status: "anemia",
          testDate: date("2025-01-12"),
          testCode: "718-7"
        })
    """)
    
    session.run("""
        MATCH (p:Patient {id: "user_4"}), 
              (d1:Disease {id: "d1_p4"}),
              (m1:Medication {id: "m1_p4"}), (m2:Medication {id: "m2_p4"}), 
              (m3:Medication {id: "m3_p4"}), (m4:Medication {id: "m4_p4"}),
              (l1:LabTest {id: "l1_p4"}), (l2:LabTest {id: "l2_p4"}), 
              (l3:LabTest {id: "l3_p4"}), (l4:LabTest {id: "l4_p4"}), (l5:LabTest {id: "l5_p4"})
        CREATE (p)-[:HAS_DISEASE]->(d1)
        CREATE (d1)-[:TREATED_BY]->(m1)
        CREATE (d1)-[:TREATED_BY]->(m2)
        CREATE (d1)-[:TREATED_BY]->(m3)
        CREATE (d1)-[:TREATED_BY]->(m4)
        CREATE (p)-[:PRESCRIBED]->(m1)
        CREATE (p)-[:PRESCRIBED]->(m2)
        CREATE (p)-[:PRESCRIBED]->(m3)
        CREATE (p)-[:PRESCRIBED]->(m4)
        CREATE (d1)-[:HAS_LAB_RESULT]->(l1)
        CREATE (d1)-[:HAS_LAB_RESULT]->(l2)
        CREATE (d1)-[:HAS_LAB_RESULT]->(l3)
        CREATE (d1)-[:HAS_LAB_RESULT]->(l4)
        CREATE (d1)-[:HAS_LAB_RESULT]->(l5)
    """)
    
    # Wearables for Patient 4
    session.run("""
        CREATE (wm1:WearableMetric {
          id: "wm1_p4",
          type: "blood_pressure",
          name: "Blood Pressure",
          unit: "mmHg",
          normalRange: "< 130/80"
        })
        CREATE (wm2:WearableMetric {
          id: "wm2_p4",
          type: "weight",
          name: "Body Weight",
          unit: "kg",
          normalRange: "stable"
        })
        CREATE (wm3:WearableMetric {
          id: "wm3_p4",
          type: "fluid_intake",
          name: "Fluid Intake",
          unit: "mL",
          normalRange: "1500-2000"
        })
        CREATE (wm4:WearableMetric {
          id: "wm4_p4",
          type: "steps",
          name: "Daily Steps",
          unit: "steps",
          normalRange: "> 5000"
        })
    """)
    
    session.run("""
        MATCH (p:Patient {id: "user_4"}), 
              (wm1:WearableMetric {id: "wm1_p4"}), (wm2:WearableMetric {id: "wm2_p4"}),
              (wm3:WearableMetric {id: "wm3_p4"}), (wm4:WearableMetric {id: "wm4_p4"})
        CREATE (p)-[:HAS_METRIC]->(wm1)
        CREATE (p)-[:HAS_METRIC]->(wm2)
        CREATE (p)-[:HAS_METRIC]->(wm3)
        CREATE (p)-[:HAS_METRIC]->(wm4)
    """)
    
    session.run("""
        CREATE (r1:Reading {id: "r_bp_1_p4", value: "135/85", timestamp: datetime("2026-02-08T08:00:00Z")})
        CREATE (r2:Reading {id: "r_bp_2_p4", value: "132/82", timestamp: datetime("2026-02-09T08:00:00Z")})
        CREATE (r3:Reading {id: "r_wt_1_p4", value: 72.5, timestamp: datetime("2026-02-08T07:00:00Z")})
        CREATE (r4:Reading {id: "r_wt_2_p4", value: 72.8, timestamp: datetime("2026-02-09T07:00:00Z")})
        CREATE (r5:Reading {id: "r_fl_1_p4", value: 1650, timestamp: datetime("2026-02-08T23:59:00Z")})
        CREATE (r6:Reading {id: "r_fl_2_p4", value: 1720, timestamp: datetime("2026-02-09T23:59:00Z")})
        CREATE (r7:Reading {id: "r_steps_1_p4", value: 4234, timestamp: datetime("2026-02-08T23:59:00Z")})
        CREATE (r8:Reading {id: "r_steps_2_p4", value: 5100, timestamp: datetime("2026-02-09T23:59:00Z")})
    """)
    
    session.run("""MATCH (wm:WearableMetric {id: "wm1_p4"}), (r:Reading) WHERE r.id IN ["r_bp_1_p4", "r_bp_2_p4"] CREATE (wm)-[:RECORDED_AS]->(r)""")
    session.run("""MATCH (wm:WearableMetric {id: "wm2_p4"}), (r:Reading) WHERE r.id IN ["r_wt_1_p4", "r_wt_2_p4"] CREATE (wm)-[:RECORDED_AS]->(r)""")
    session.run("""MATCH (wm:WearableMetric {id: "wm3_p4"}), (r:Reading) WHERE r.id IN ["r_fl_1_p4", "r_fl_2_p4"] CREATE (wm)-[:RECORDED_AS]->(r)""")
    session.run("""MATCH (wm:WearableMetric {id: "wm4_p4"}), (r:Reading) WHERE r.id IN ["r_steps_1_p4", "r_steps_2_p4"] CREATE (wm)-[:RECORDED_AS]->(r)""")
    
    print("✓ Patient 4 complete with all nodes")
    
    # ============================================================
    # PATIENT 5: Robert Brown - Multiple Conditions
    # (Type 2 Diabetes + Hypertension + Heart Disease)
    # ============================================================
    print("\n[6/6] Creating Patient 5: Robert Brown (Multiple Conditions)...")
    
    session.run("""
        CREATE (p:Patient {
          id: "user_5",
          name: "Robert Brown",
          age: 71,
          gender: "Male",
          bloodType: "O-",
          created_at: datetime()
        })
        
        CREATE (d1:Disease {
          id: "d1_p5",
          name: "Type 2 Diabetes",
          category: "Endocrine",
          severity: "severe",
          status: "active",
          diagnosisDate: date("2012-04-10"),
          icd10: "E11.9"
        })
        
        CREATE (d2:Disease {
          id: "d2_p5",
          name: "Hypertension",
          category: "Cardiovascular",
          severity: "moderate",
          status: "active",
          diagnosisDate: date("2010-07-22"),
          icd10: "I10"
        })
        
        CREATE (d3:Disease {
          id: "d3_p5",
          name: "Heart Disease",
          category: "Cardiovascular",
          severity: "moderate",
          status: "active",
          diagnosisDate: date("2019-02-15"),
          icd10: "I25.10"
        })
        
        CREATE (m1:Medication {
          id: "m1_p5",
          name: "Insulin Glargine",
          dosage: "20 units",
          frequency: "once daily",
          purpose: "Blood sugar control",
          generic: false,
          atcCode: "A10AE04"
        })
        
        CREATE (m2:Medication {
          id: "m2_p5",
          name: "Metformin",
          dosage: "1000mg",
          frequency: "twice daily",
          purpose: "Blood sugar control",
          generic: true,
          atcCode: "A10BA02"
        })
        
        CREATE (m3:Medication {
          id: "m3_p5",
          name: "Amlodipine",
          dosage: "10mg",
          frequency: "once daily",
          purpose: "Blood pressure control",
          generic: true,
          atcCode: "C08CA01"
        })
        
        CREATE (m4:Medication {
          id: "m4_p5",
          name: "Atorvastatin",
          dosage: "80mg",
          frequency: "once daily",
          purpose: "Cholesterol management",
          generic: true,
          atcCode: "C10AA05"
        })
        
        CREATE (m5:Medication {
          id: "m5_p5",
          name: "Aspirin",
          dosage: "81mg",
          frequency: "once daily",
          purpose: "Blood thinner",
          generic: true,
          atcCode: "B01AC06"
        })
        
        CREATE (m6:Medication {
          id: "m6_p5",
          name: "Carvedilol",
          dosage: "25mg",
          frequency: "twice daily",
          purpose: "Heart failure management",
          generic: true,
          atcCode: "C07AG02"
        })
        
        CREATE (l1:LabTest {
          id: "l1_p5",
          name: "HbA1c",
          result: 8.5,
          unit: "%",
          normalRange: "< 5.7%",
          status: "poorly controlled",
          testDate: date("2025-01-15"),
          testCode: "2345-7"
        })
        
        CREATE (l2:LabTest {
          id: "l2_p5",
          name: "Fasting Glucose",
          result: 185,
          unit: "mg/dL",
          normalRange: "70-100 mg/dL",
          status: "high",
          testDate: date("2025-01-15"),
          testCode: "1558-6"
        })
        
        CREATE (l3:LabTest {
          id: "l3_p5",
          name: "BNP",
          result: 280,
          unit: "pg/mL",
          normalRange: "< 100 pg/mL",
          status: "elevated",
          testDate: date("2025-01-15"),
          testCode: "35637-4"
        })
        
        CREATE (l4:LabTest {
          id: "l4_p5",
          name: "LDL Cholesterol",
          result: 95,
          unit: "mg/dL",
          normalRange: "< 100 mg/dL",
          status: "near target",
          testDate: date("2025-01-15"),
          testCode: "18262-6"
        })
        
        CREATE (l5:LabTest {
          id: "l5_p5",
          name: "Blood Pressure",
          result: "148/92",
          unit: "mmHg",
          normalRange: "< 120/80 mmHg",
          status: "stage 2 hypertension",
          testDate: date("2025-01-15"),
          testCode: "55284-4"
        })
    """)
    
    session.run("""
        MATCH (p:Patient {id: "user_5"}), 
              (d1:Disease {id: "d1_p5"}), (d2:Disease {id: "d2_p5"}), (d3:Disease {id: "d3_p5"}),
              (m1:Medication {id: "m1_p5"}), (m2:Medication {id: "m2_p5"}), 
              (m3:Medication {id: "m3_p5"}), (m4:Medication {id: "m4_p5"}), 
              (m5:Medication {id: "m5_p5"}), (m6:Medication {id: "m6_p5"}),
              (l1:LabTest {id: "l1_p5"}), (l2:LabTest {id: "l2_p5"}), 
              (l3:LabTest {id: "l3_p5"}), (l4:LabTest {id: "l4_p5"}), (l5:LabTest {id: "l5_p5"})
        CREATE (p)-[:HAS_DISEASE]->(d1)
        CREATE (p)-[:HAS_DISEASE]->(d2)
        CREATE (p)-[:HAS_DISEASE]->(d3)
        CREATE (d1)-[:TREATED_BY]->(m1)
        CREATE (d1)-[:TREATED_BY]->(m2)
        CREATE (d2)-[:TREATED_BY]->(m3)
        CREATE (d3)-[:TREATED_BY]->(m4)
        CREATE (d3)-[:TREATED_BY]->(m5)
        CREATE (d3)-[:TREATED_BY]->(m6)
        CREATE (p)-[:PRESCRIBED]->(m1)
        CREATE (p)-[:PRESCRIBED]->(m2)
        CREATE (p)-[:PRESCRIBED]->(m3)
        CREATE (p)-[:PRESCRIBED]->(m4)
        CREATE (p)-[:PRESCRIBED]->(m5)
        CREATE (p)-[:PRESCRIBED]->(m6)
        CREATE (d1)-[:HAS_LAB_RESULT]->(l1)
        CREATE (d1)-[:HAS_LAB_RESULT]->(l2)
        CREATE (d2)-[:HAS_LAB_RESULT]->(l5)
        CREATE (d3)-[:HAS_LAB_RESULT]->(l3)
        CREATE (d3)-[:HAS_LAB_RESULT]->(l4)
    """)
    
    # Wearables for Patient 5
    session.run("""
        CREATE (wm1:WearableMetric {
          id: "wm1_p5",
          type: "blood_glucose",
          name: "Blood Glucose",
          unit: "mg/dL",
          normalRange: "70-130"
        })
        CREATE (wm2:WearableMetric {
          id: "wm2_p5",
          type: "blood_pressure",
          name: "Blood Pressure",
          unit: "mmHg",
          normalRange: "< 130/80"
        })
        CREATE (wm3:WearableMetric {
          id: "wm3_p5",
          type: "heart_rate",
          name: "Heart Rate",
          unit: "bpm",
          normalRange: "60-100"
        })
        CREATE (wm4:WearableMetric {
          id: "wm4_p5",
          type: "weight",
          name: "Body Weight",
          unit: "kg",
          normalRange: "stable"
        })
        CREATE (wm5:WearableMetric {
          id: "wm5_p5",
          type: "steps",
          name: "Daily Steps",
          unit: "steps",
          normalRange: "> 5000"
        })
    """)
    
    session.run("""
        MATCH (p:Patient {id: "user_5"}), 
              (wm1:WearableMetric {id: "wm1_p5"}), (wm2:WearableMetric {id: "wm2_p5"}),
              (wm3:WearableMetric {id: "wm3_p5"}), (wm4:WearableMetric {id: "wm4_p5"}),
              (wm5:WearableMetric {id: "wm5_p5"})
        CREATE (p)-[:HAS_METRIC]->(wm1)
        CREATE (p)-[:HAS_METRIC]->(wm2)
        CREATE (p)-[:HAS_METRIC]->(wm3)
        CREATE (p)-[:HAS_METRIC]->(wm4)
        CREATE (p)-[:HAS_METRIC]->(wm5)
    """)
    
    session.run("""
        CREATE (r1:Reading {id: "r_bg_1_p5", value: 178, timestamp: datetime("2026-02-08T08:00:00Z")})
        CREATE (r2:Reading {id: "r_bg_2_p5", value: 192, timestamp: datetime("2026-02-09T08:00:00Z")})
        CREATE (r3:Reading {id: "r_bp_1_p5", value: "145/90", timestamp: datetime("2026-02-08T08:00:00Z")})
        CREATE (r4:Reading {id: "r_bp_2_p5", value: "148/92", timestamp: datetime("2026-02-09T08:00:00Z")})
        CREATE (r5:Reading {id: "r_hr_1_p5", value: 78, timestamp: datetime("2026-02-08T08:00:00Z")})
        CREATE (r6:Reading {id: "r_hr_2_p5", value: 82, timestamp: datetime("2026-02-09T08:00:00Z")})
        CREATE (r7:Reading {id: "r_wt_1_p5", value: 88.2, timestamp: datetime("2026-02-08T07:00:00Z")})
        CREATE (r8:Reading {id: "r_wt_2_p5", value: 88.5, timestamp: datetime("2026-02-09T07:00:00Z")})
        CREATE (r9:Reading {id: "r_steps_1_p5", value: 3234, timestamp: datetime("2026-02-08T23:59:00Z")})
        CREATE (r10:Reading {id: "r_steps_2_p5", value: 4100, timestamp: datetime("2026-02-09T23:59:00Z")})
    """)
    
    session.run("""MATCH (wm:WearableMetric {id: "wm1_p5"}), (r:Reading) WHERE r.id IN ["r_bg_1_p5", "r_bg_2_p5"] CREATE (wm)-[:RECORDED_AS]->(r)""")
    session.run("""MATCH (wm:WearableMetric {id: "wm2_p5"}), (r:Reading) WHERE r.id IN ["r_bp_1_p5", "r_bp_2_p5"] CREATE (wm)-[:RECORDED_AS]->(r)""")
    session.run("""MATCH (wm:WearableMetric {id: "wm3_p5"}), (r:Reading) WHERE r.id IN ["r_hr_1_p5", "r_hr_2_p5"] CREATE (wm)-[:RECORDED_AS]->(r)""")
    session.run("""MATCH (wm:WearableMetric {id: "wm4_p5"}), (r:Reading) WHERE r.id IN ["r_wt_1_p5", "r_wt_2_p5"] CREATE (wm)-[:RECORDED_AS]->(r)""")
    session.run("""MATCH (wm:WearableMetric {id: "wm5_p5"}), (r:Reading) WHERE r.id IN ["r_steps_1_p5", "r_steps_2_p5"] CREATE (wm)-[:RECORDED_AS]->(r)""")
    
    print("✓ Patient 5 complete with all nodes")
    
    # Verify the setup
    print("\n" + "=" * 60)
    print("VERIFICATION - ALL PATIENTS")
    print("=" * 60)
    
    for patient_id in ["user_1", "user_2", "user_3", "user_4", "user_5"]:
        result = session.run("""
            MATCH (p:Patient {id: $patient_id})
            OPTIONAL MATCH (p)-[:HAS_DISEASE]->(d:Disease)
            OPTIONAL MATCH (d)-[:TREATED_BY]->(m:Medication)
            OPTIONAL MATCH (p)-[:PRESCRIBED]->(pm:Medication)
            OPTIONAL MATCH (d)-[:HAS_LAB_RESULT]->(l:LabTest)
            OPTIONAL MATCH (p)-[:HAS_METRIC]->(wm:WearableMetric)
            OPTIONAL MATCH (wm)-[:RECORDED_AS]->(r:Reading)
            RETURN
              p.name AS patient,
              p.age AS age,
              collect(DISTINCT d.name) AS diseases,
              collect(DISTINCT pm.name) AS medications,
              collect(DISTINCT l.name) AS labs,
              collect(DISTINCT wm.name) AS wearables,
              count(DISTINCT r) AS reading_count
        """, patient_id=patient_id)
        
        record = result.single()
        if record:
            print(f"\n✓ Patient: {record['patient']} (Age: {record['age']})")
            print(f"  - Diseases: {', '.join([d for d in record['diseases'] if d])}")
            print(f"  - Medications: {', '.join([m for m in record['medications'] if m])}")
            print(f"  - Lab tests: {len([l for l in record['labs'] if l])}")
            print(f"  - Wearable metrics: {len([w for w in record['wearables'] if w])}")
            print(f"  - Total readings: {record['reading_count']}")
    
    # Count total data
    result = session.run("MATCH (n) RETURN count(n) as count")
    total = result.single()["count"]
    print(f"\n{'=' * 60}")
    print(f"✓ Total nodes created: {total}")
    
    # Count by type
    result = session.run("""
        MATCH (n)
        RETURN labels(n)[0] AS label, count(n) AS count
        ORDER BY count DESC
    """)
    print(f"\nNode breakdown:")
    for record in result:
        print(f"  - {record['label']}: {record['count']}")

driver.close()

print("\n" + "=" * 60)
print("✓ MULTI-PATIENT NEO4J SETUP COMPLETE!")
print("=" * 60)
print("\nPatient Summary:")
print("  1. John Doe (52) - Type 2 Diabetes + Hypertension")
print("  2. Sarah Smith (58) - Heart Disease")
print("  3. Michael Johnson (34) - Asthma")
print("  4. Emily Davis (64) - Chronic Kidney Disease")
print("  5. Robert Brown (71) - Type 2 Diabetes + Hypertension + Heart Disease")
print("\nAll patients have complete medical profiles including:")
print("  ✓ Diseases with ICD-10 codes")
print("  ✓ Medications with dosages and ATC codes")
print("  ✓ Lab test results")
print("  ✓ Wearable metrics and readings")
print("=" * 60)