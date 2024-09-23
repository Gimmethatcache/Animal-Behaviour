from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import *
from PyQt5 import uic
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class LoginThread(QThread):
    connection_established = pyqtSignal(bool)

    def __init__(self, server_address, server_port, username, password):
        super(LoginThread, self).__init__()
        self.server_address = server_address
        self.server_port = server_port
        self.username = username
        self.password = password
        self.server = None  # Initialize the server object

    def run(self):
        try:
            server = smtplib.SMTP(self.server_address, self.server_port)
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(self.username, self.password)
            self.server = server  # Save the server object
            self.connection_established.emit(True)
        except Exception as e:
            print("Error:", e)
            self.connection_established.emit(False)

class MyGUI(QMainWindow):
    def __init__(self):
        super(MyGUI, self).__init__()
        uic.loadUi(r"C:\Users\ksvib\Downloads\Animal Behaviour Prediction ~ Source Codes\Mail\Mail.ui", self)
        self.show()

        self.pushButton.clicked.connect(self.login)
        self.pushButton_2.clicked.connect(self.attach_sth)
        self.pushButton_3.clicked.connect(self.send_mail)

        # Disable buttons initially
        self.disable_buttons()

    def disable_buttons(self):
        self.lineEdit_5.setEnabled(False)
        self.lineEdit_6.setEnabled(False)
        self.textEdit.setEnabled(False)
        self.pushButton_2.setEnabled(False)
        self.pushButton_3.setEnabled(False)

    def enable_buttons(self):
        self.lineEdit_5.setEnabled(True)
        self.lineEdit_6.setEnabled(True)
        self.textEdit.setEnabled(True)
        self.pushButton_2.setEnabled(True)
        self.pushButton_3.setEnabled(True)

    def login(self):
        server_address = self.lineEdit_3.text()
        server_port = self.lineEdit_4.text()
        username = self.lineEdit.text()
        password = self.lineEdit_2.text()

        self.login_thread = LoginThread(server_address, server_port, username, password)
        self.login_thread.connection_established.connect(self.handle_login_result)
        self.login_thread.start()

        # Disable login button during login attempt
        self.pushButton.setEnabled(False)

    def handle_login_result(self, success):
        if success:
            print("Login successful")
            # Enable other input fields and buttons
            self.enable_buttons()
        else:
            # Re-enable login button if login failed
            print("Login failed")
            self.pushButton.setEnabled(True)
            # Display error message if login failed
            message_box = QMessageBox()
            message_box.setText("Failed to establish connection.")
            message_box.exec()

    def attach_sth(self):
        pass

    def send_mail(self):
        dialog = QMessageBox()
        dialog.setText("Do you want to send this mail?")
        dialog.addButton(QPushButton("Yes"), QMessageBox.YesRole) #0
        dialog.addButton(QPushButton("No"), QMessageBox.NoRole) #1

        if dialog.exec_() == 0:
            try:
                if self.login_thread.server:  # Check if server object exists and is initialized
                    print("SMTP server initialized.")
                    self.msg = MIMEMultipart()
                    self.msg['From'] = "Home"
                    self.msg['To'] = self.lineEdit_5.text()
                    self.msg['Subject'] = self.lineEdit_6.text()
                    textEdit = self.findChild(QTextEdit, "textEdit")
                    if textEdit:
                        self.msg.attach(MIMEText(textEdit.toPlainText(), 'plain'))
                        text = self.msg.as_string()
                        self.login_thread.server.sendmail(self.lineEdit.text(), self.lineEdit_5.text(), text)
                        message_box = QMessageBox()
                        message_box.setText("Mail Sent!")
                        message_box.exec()
                    else:
                        raise Exception("textEdit widget not found.")
                else:
                    print("SMTP server not initialized.")
                    raise Exception("SMTP server not initialized.")
            except Exception as e:
                print("Error:", e)
                message_box = QMessageBox()
                message_box.setText("Sending Mail Failed!")
                message_box.exec()


app = QApplication([])
Window = MyGUI()
app.exec_()
