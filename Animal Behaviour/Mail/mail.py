from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from PyQt5 import uic

import smtplib
from email import encoders
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart


class MyGUI(QMainWindow):

    def __init__(self):
        super(MyGUI, self).__init__()
        uic.loadUi(r"C:\Users\ksvib\Downloads\Animal Behaviour Prediction ~ Source Codes\Mail\Mail.ui",self)
        self.show()

        self.pushButton.clicked.connect(self.login)
        self.pushButton_2.clicked.connect(self.attach_sth)
        self.pushButton_3.clicked.connect(self.send_mail)
    
    def login(self):
        # print("HELLO")
        try:
            self.server = smtplib.SMTP(self.lineEdit_3.text(), self.lineEdit_4.text())
            self.server.ehlo()
            self.server.starttls()
            self.server.ehlo()
            self.server.login(self.lineEdit.text(),self.lineEdit_2.text())
            print("Connection Established!!")

            self.lineEdit.setEnabled(False)
            self.lineEdit_2.setEnabled(False)
            self.lineEdit_3.setEnabled(False)
            self.lineEdit_4.setEnabled(False)
            self.pushButton.setEnabled(False)

            self.lineEdit_5.setEnabled(True)
            self.lineEdit_6.setEnabled(True)
            self.lineEdit_7.setEnabled(True)
            self.pushButton_2.setEnabled(True)
            self.pushButton_3.setEnabled(True)

            self.msg = MIMEMultipart()
        except smtplib.SMTPAuthenticationError:
            message_box = QMessageBox()
            message_box.setText("Invalid Login Info!")
            message_box.exec()
        except:
            message_box = QMessageBox()
            message_box.setText("Login Failed!")
            message_box.exec()
            
    def attach_sth(self):
        pass

    def send_mail(self):
        pass

app = QApplication([])
Window = MyGUI()
app.exec_()