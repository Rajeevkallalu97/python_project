
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

import os


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
    def change(self,name):
        self.manager.current = name
    def getContents(self):
        f = open(self.filename, "r")
        if f.mode == 'r':
            contents = f.read()
        for i in range(0,5):
            print(contents)


class ContentNavigationDrawer(BoxLayout):
    screen_manager = ObjectProperty()
    nav_drawer = ObjectProperty()   

class Researcher(Screen):
    pass

class Login(Screen):

    def gettoast(self,StringProperty):
        toast(StringProperty)

    def check(self, username, password):
        connect = mysql.connector.connect(**config)
        cursor = connect.cursor()
        username1 = (username,)
        id_query = "SELECT id FROM user_login WHERE id = %s"
        cursor.execute(id_query,username1)
        id = cursor.fetchall()
        if not id:
            self.gettoast("Invalid Username")
        else:
            password = (str(username),password)
            password_query = "SELECT password FROM user_login WHERE id = %s AND password = %s"
            cursor.execute(password_query,(password))
            password = cursor.fetchall()
            if not password:
                self.gettoast("Invalid password")
            else:
                self.change()
    def change(self):
        self.manager.current = 'scr 7'


class Plot(Screen):
    graph_test = ObjectProperty(None)
    def come(self):
        plot = MeshLinePlot(color=[1, 0, 0, 1])
        plot.points = [(x, sin(x / 10.)) for x in range(0, 101)]
        self.graph_test.add_plot(plot)
    
     
class piSO2(MDApp):

    def build(self):
        self.theme_cls.primary_palette = "Gray"
        return Builder.load_file("main_design.kv")

piSO2().run()