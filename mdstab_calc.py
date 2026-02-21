# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 17:33:43 2025

@author: sdavilao
"""
import numpy as np
import pandas as pd

Sc = 1.25  
pw = 1000  
ps = 1600  
g = 9.81  
yw = g * pw
ys = g * ps
phi = np.deg2rad(41)  
m = 1  
l = 10  
w = 6.7  
C0 = 760  
j = 0.8 

hollow_rad = np.radians(38.27)
z = 2.763

Crb = C0 * np.exp(-z * j)
Crl = (C0 / (j * z)) * (1 - np.exp(-z * j))

K0 = 1 - np.sin(hollow_rad)

Kp = np.tan((np.deg2rad(45))+(phi/2))**2
Ka = np.tan((np.deg2rad(45))-(phi/2))**2


Frb = (Crb + ((np.cos(hollow_rad)) ** 2) * z * (ys - yw * m) * np.tan(phi)) * l * w
Frc = (Crl + (K0 * 0.5 * z * (ys - yw * m ** 2) * np.tan(phi))) * (np.cos(hollow_rad) * z * l * 2)
Frddu = (Kp - Ka) * 0.5 * (z ** 2) * (ys - yw * (m ** 2)) * w
Fdc = (np.sin(hollow_rad)) * (np.cos(hollow_rad)) * z * ys * l * w

(Frb + Frc + Frddu) / Fdc 