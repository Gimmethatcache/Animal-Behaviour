import sys
from PyQt5.QtWidgets import QApplication, QDialog, QStackedWidget, QMainWindow, QStyle, QMessageBox
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt
import mysql.connector as con
from Main import ModelDeployer

class CenteredMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

    def changeEvent(self, event):
        if event.type() == event.WindowStateChange:
            if self.windowState() & Qt.WindowMaximized:
                self.resize(self.size() * 1.5)  # Increase size by 50% when maximized
            else:
                self.setGeometry(
                    QStyle.alignedRect(
                        Qt.LeftToRight,
                        Qt.AlignCenter,
                        self.size(),
                        QApplication.desktop().availableGeometry()
                    )
                )
        super().changeEvent(event)

class LoginApp(QDialog):
    def __init__(self):
        super(LoginApp, self).__init__()
        loadUi("login-form.ui", self)
        self.b1.clicked.connect(self.login)
        self.b2.clicked.connect(self.show_reg)

    def login(self):
        un = self.tb1.text()
        pw = self.tb2.text()
        db = con.connect(host="localhost",user="root",password="",db="sample")
        cursor = db.cursor()
        cursor.execute("select * from userlist where username='"+un+"' and password ='"+ pw +"'")
        result = cursor.fetchone()
        self.tb1.setText("")
        self.tb2.setText("")
        if result:
            QMessageBox.information(self,"Login Output","Congrats!! You have logged in Successfully")
            # Close the login window
            self.close()
            # Open the ModelDeployer window
            self.open_main_window()
        else:
            QMessageBox.information(self, "Login Output", "Invalid User!! Register For New User!!")

    def show_reg(self):
        widget.setCurrentIndex(1)

    def open_main_window(self):
        self.main_window = ModelDeployer()
        self.main_window.show()



class RegApp(QDialog):
    def __init__(self):
        super(RegApp, self).__init__()
        loadUi("Register-form.ui", self)
        self.b3.clicked.connect(self.reg)
        self.b4.clicked.connect(self.show_login)

    def reg(self):
        un = self.tb3.text()
        pw = self.tb4.text()
        em = self.tb5.text()
        ph = self.tb6.text()

        db = con.connect(host="localhost", user="root", password="", db="sample")

        cursor = db.cursor()
        cursor.execute("SELECT * FROM userlist WHERE username = %s AND password = %s", (un, pw))
        result = cursor.fetchone()
        if result:
            QMessageBox.information(self, "Login form", "The User already registered, Try another username!!")
        else:
            cursor.execute("INSERT INTO userlist (username, password, email, phone) VALUES (%s, %s, %s, %s)",
                           (un, pw, em, ph))
            db.commit()
            QMessageBox.information(self, "Login form", "The user Registered Successfully,You can Login now!!")
    def show_login(self):
            widget.setCurrentIndex(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    main_window = CenteredMainWindow()
    widget = QStackedWidget(main_window)

    login_form = LoginApp()
    registration_form = RegApp()

    widget.addWidget(login_form)
    widget.addWidget(registration_form)
    widget.setCurrentIndex(0)

    main_window.setCentralWidget(widget)
    main_window.setWindowTitle("Your Application Title")
    main_window.resize(400, 500)
    main_window.show()

    sys.exit(app.exec_())
