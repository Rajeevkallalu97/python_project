
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
import matplotlib.pyplot as plt
import os
import plot as p 

config = {
  'user': 'root',
  'password': '',
  'host': '127.0.0.1',
  'database': 'project',
  'raise_on_warnings': True
}

class ScreenManagement(ScreenManager):
    pass


class FilePath(Screen):
    def change(self,name):
        self.manager.current = name
    def selected(self, path, filename):
        FileSelection.path = filename
        FileSelection.filename = filename[0]
        print(filename)

class FileSelection(Screen):
    path = ""
    filename = ""
    def transition(self):
        if not Plot.filename:
            Login.gettoast("Please Select a file")
        else:
            self.change('scr 2')
    def change(self,name):
        self.manager.current = name
    def getContents(self):
        Plot.filename = self.filename

class ContentNavigationDrawer(BoxLayout):
    screen_manager = ObjectProperty()
    nav_drawer = ObjectProperty()   

class Researcher(Screen):
    pass

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


class Plot(Screen):
    filename = ""
    current = 0
    current_modified = 0
    harmonic = 0
    def plot(self):
        if not self.filename:
            Login.gettoast("Please Select a File")
        else:
            p.withFile(self.filename,self.current,self.current_modified,self.harmonic)

class piSO2(MDApp):

    def build(self):
        self.theme_cls.primary_palette = "Gray"
        return Builder.load_file("main_design.kv")

piSO2().run()