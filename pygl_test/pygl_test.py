#!/usr/bin/env python

from PyQt5 import QtGui, QtWidgets, uic
from PyQt5.QtWidgets import (QFileDialog, QOpenGLWidget)
from PyQt5.QtCore import QThread


class WfWidget(QOpenGLWidget):
    def __init__(self, parent = None):
        super(WfWidget, self).__init__(parent)

    def paintGL(self):
        self.gl.glColor3f(0.0, 0.0, 1.0)
        self.gl.glRectf(-5, -5, 5, 5)
        self.gl.glColor3f(1.0, 0.0, 0.0)
        self.gl.glBegin(self.gl.GL_LINES)
        self.gl.glVertex3f(0, 0, 0)
        self.gl.glVertex3f(20, 20, 0)
        self.gl.glEnd()

    def resizeGL(self, w, h):
        self.gl.glMatrixMode(self.gl.GL_PROJECTION)
        self.gl.glLoadIdentity()
        self.gl.glOrtho(-50, 50, -50, 50, -50.0, 50.0)
        self.gl.glViewport(0, 0, w, h)

    def initializeGL(self):
        self.gl = self.context().versionFunctions()
        self.gl.initializeOpenGLFunctions()

        self.gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        self.gl.glClear(self.gl.GL_COLOR_BUFFER_BIT)

if __name__ == '__main__':
    app = QtWidgets.QApplication(["Winfred's PyQt OpenGL"])
    widget = WfWidget()
    widget.show()
