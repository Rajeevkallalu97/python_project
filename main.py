import math
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ObjectProperty, StringProperty
from kivymd.app import MDApp
from kivymd.theming import ThemableBehavior
from kivymd.uix.list import OneLineIconListItem, MDList
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen
from kivymd.uix.filemanager import MDFileManager
from kivymd.toast import toast
from kivy_garden.graph import Graph, MeshLinePlot
from math import sin
import mysql.connector
from mysql.connector import errorcode
from kivy.uix.checkbox import CheckBox
import matplotlib.pyplot as plt
import os
import plot as p 
import excitation as ext


config = {
  'user': 'root',
  'password': '',
  'host': '127.0.0.1',
  'database': 'project',
  'raise_on_warnings': True
}

error_text = "Something went wrong please try again"
calib_text_high = "High concentration, out of calibration range"
calib_text_normal = "Calibration range normal"
peak_area = error_text
peak_height = error_text
abs_peak_height = error_text
calib_text = "No Results to show"
concentration = error_text

class ScreenManagement(ScreenManager):
    pass


#Login class for researcher
class Login(Screen):

    def gettoast(StringProperty):
        toast(StringProperty)

    def check(self, username, password):
        connect = mysql.connector.connect(**config)
        cursor = connect.cursor()
        username1 = (username,)
        id_query = "SELECT id FROM user_login WHERE id = %s"
        cursor.execute(id_query,username1)
        id = cursor.fetchall()
        if not id:
            Login.gettoast("Invalid Username")
        else:
            password = (str(username),password)
            password_query = "SELECT password FROM user_login WHERE id = %s AND password = %s"
            cursor.execute(password_query,(password))
            password = cursor.fetchall()
            if not password:
                Login.gettoast("Invalid password")
            else:
                self.change()
    def change(self):
        self.manager.current = 'scr 7'


#This class is for citizen login
class Citizen(Screen):
    filename = StringProperty()
    def transition(self):
        if not Plot.filename:
            Login.gettoast("Please Select a file")
        else:
            global peak_area, peak_height, abs_peak_height, calib_text, concentration
            file = open(self.filename,"r")
            for line in file:
                fields = line.split(",")
            file.close()
            #Parameters of files are as stable_duration, sample rate, v1, v2, v3, frequency, duration, abc constants
            result = ext.temp_call_citizen(float(fields[0]),int(fields[1]),float(fields[2]),float(fields[3]),
            float(fields[4]),float(fields[5]),float(fields[6]))
            peak_area = str(result[1])
            peak_height = str(result[0])
            abs_peak_height = str(result[2])
            #Checking claibration
            a = float(fields[7])
            b = float(fields[8])
            c = float(fields[9])
            value = result[0]
            val1 = 2-4*a*(c-value)
            conc = (-b+math.sqrt(b ** val1))/(2*a)
            concentration = str(conc)+" ppm Free SO2 "
            if conc > (-b/2*a)*0.85:
                calib_text = calib_text_high
                print(calib_text_high)
            else:
                calib_text = calib_text_normal
                print(calib_text_normal)
            self.change('scr 4')

    def change(self,name):
        self.manager.current = name
    def getContents(self):
        Plot.filename = self.filename

#This class is for researcher login
class Researcher(Screen):
    filename = StringProperty()
    def transition(self):
        if not Plot.filename:
            Login.gettoast("Please Select a file")
        else:
            global peak_area, peak_height, abs_peak_height, calib_text, concentration
            file = open(self.filename,"r")
            for line in file:
                fields = line.split(",")
            file.close()
            result = ext.temp_call_researcher(float(fields[0]),int(fields[1]),float(fields[2]),float(fields[3]),
            float(fields[4]),float(fields[5]),float(fields[6]))
            peak_area = str(result[1])
            peak_height = str(result[0])
            abs_peak_height = str(result[2])
            #Checking claibration
            a = float(fields[7])
            b = float(fields[8])
            c = float(fields[9])
            value = result[0]
            val1 = 2-4*a*(c-value)
            conc = (-b+math.sqrt(b ** val1))/(2*a)
            concentration = str(conc)+" ppm Free SO2 "
            if conc > (-b/2*a)*0.85:
                calib_text = calib_text_high
                print(calib_text_high)
            else:
                calib_text = calib_text_normal
                print(calib_text_normal)
            self.change('scr 4')

    def parameter(self,amplititude,sample,freq,stable,record,v1,v2,v3,a,b,c):
        #print("ampli",amplititude,"ssample",sample,"freq",freq,"record",record,"v1",v1,v2,v3,a,b,c)
        global peak_area, peak_height, abs_peak_height, calib_text, concentration
        
        #Parameters of files are as stable_duration, sample rate, v1, v2, v3, frequency, duration, abc constants
        result = ext.temp_call_researcher_amp(float(stable),float(amplititude),int(sample),float(record),float(freq),float(v1),float(v2),float(v3))
       
        peak_area = str(result[1])
        peak_height = str(result[0])
        abs_peak_height = str(result[2])
        #Checking claibration
        a = float(a)
        b = float(b)
        c = float(c)
        value = result[0]
        val1 = 2-4*a*(c-value)
        conc = (-b+math.sqrt(b ** val1))/(2*a)
        concentration = str(conc)+" ppm Free SO2 "
        if conc > (-b/2*a)*0.85:
                calib_text = calib_text_high
                print(calib_text_high)
        else:
            calib_text = calib_text_normal
            print(calib_text_normal)
        self.change('scr 4')
    def change(self,name):
        self.manager.current = name
    def getContents(self):
        Plot.filename = self.filename

#This class is to show results from either of the login depending on the states
class Results(Screen):
    def get(self):
        self.ids.area.text = peak_area
        self.ids.height.text = peak_height
        self.ids.abs.text = abs_peak_height
        self.ids.calib.text = calib_text
        self.ids.concentration.text = concentration
    

class ContentNavigationDrawer(BoxLayout):
    screen_manager = ObjectProperty()
    nav_drawer = ObjectProperty()   


class FilePath_Citizen(Screen):
    def change(self,name):
        self.manager.current = name
    def selected(self,filename):
        Citizen.filename = filename[0]
        print(filename)

class FilePath_Plot(Screen):
    def change(self,name):
        self.manager.current = name
    def selected(self,filename):
        Plot.filename = filename[0]
        print(filename)

class FilePath_Researcher(Screen):
    def change(self,name):
        self.manager.current = name
    def selected(self,filename):
        Researcher.filename = filename[0]
        print(filename)


class Plot(Screen):
    filename = StringProperty()
    current = 0
    current_modified = 0
    harmonic = 0
    def on_checkbox1_active(self, checkbox, value):
        if value:
           self.current = 1
        else:
           self.current = 0
    def on_checkbox2_active(self, checkbox, value):
        if value:
            self.current_modified = 1
        else:
            self.current_modified = 0
    def on_checkbox3_active(self, checkbox, value):
        if value:
            self.harmonic = 1
        else:
            self.harmonic = 0
    def plot(self):
        if not self.filename:
            Login.gettoast("Please Select a File")
        else:
            p.withFile(self.filename,self.current,self.current_modified,self.harmonic)
            self.current = 0
            self.current_modified = 0
            self.harmonic = 0
    def change(self,name):
        self.manager.current = name
    def getContents(self):
        Plot.filename = self.filename


class piSO2(MDApp):
    def build(self):
        self.theme_cls.primary_palette = "Gray"
        return Builder.load_file("main_design.kv")

piSO2().run()