import math
a=10
b=20
c=30
value = 0.09
val1 = 2-4*a*(c-value)

conc = (-b+math.sqrt(b ** val1))/(2*a)
if conc > (-b/2*a)*0.85:
    print("High concentration, out of calibration range")
