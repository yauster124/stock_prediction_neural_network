import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from session import Session
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import yfinance as yf
import datetime
from functools import partial
import matplotlib
from dnn import Dnn
import time

font = {'family' : 'Calibri',
        'size'   : 10,
        'weight' : 'bold'}
# Colours
ORANGE = 'rgb(255, 140, 0)'
GREY = 'rgb(64, 64, 64)'
LIGHT_ORANGE = 'rgb(255, 175, 64)'
WARM_RED = 'rgb(249, 66, 58)'
LIGHT_GREY = 'rgb(200, 200, 200)'
DARK_GREY = 'rgb(32, 32, 32)'
matplotlib.rc('font', **font)
matplotlib.rc('text', color='white')
matplotlib.rc('xtick', color='white')
matplotlib.rc('ytick', color='white')
matplotlib.rc('axes', facecolor='#404040')
matplotlib.rc('axes', grid=True)
matplotlib.rc('grid', alpha=0.5, linewidth=0.4)


class MainUi(QWidget):
    """Contains all widgets."""
    def __init__(self, session=Session()):
        super().__init__()
        self.setWindowState(Qt.WindowMaximized)
        sizeObject = QDesktopWidget().screenGeometry(-1)
        self.screen_height = sizeObject.height()
        self.screen_width = sizeObject.width()
        self.session = session
        self.stack = QStackedWidget(self)
        
        self.stack.addWidget(LoginScreen(self))
        self.stack.addWidget(MainScreen(self))
        self.stack.addWidget(ViewMyStocksScreen(self))
        self.stack.addWidget(SignupScreen(self))
        
        hbox = QHBoxLayout(self)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(self.stack)
        self.setLayout(hbox)
        self.setWindowTitle('Fortune')
        self.stack.setCurrentIndex(0)
        self.setWindowIcon(QIcon('images/logo.png'))
        self.show()

    def show_window(self, i, widget):
        self.stack.removeWidget(self.stack.widget(i))
        self.stack.insertWidget(i, widget(self))
        self.stack.setCurrentIndex(i)


class LoginScreen(QWidget):
    def __init__(self, master=None):
        super().__init__()
        self.master = master
        self.session = self.master.session
        self.master_layout = QHBoxLayout()
        self.master_layout.setContentsMargins(0, 0, 0, 0)
        self.master_container = QWidget()
        self.setStyleSheet(f'background-color: {DARK_GREY}')
        vlayout = QVBoxLayout()
        gbox = QGroupBox()
        self.layout = QFormLayout()
        self.layout.setContentsMargins(0.02*self.master.screen_width, 0.05*self.master.screen_width, 0.02*self.master.screen_width, 0.05*self.master.screen_width)
        self.titlebar = QLabel('FORTUNE')
        self.titlebar.setStyleSheet('QLabel {background: rgb(255, 140, 0); padding: 16px; color: white}')
        font = Font(30)
        self.titlebar.setFont(font)
        self.username_e = QLineEdit()
        self.username_e.setPlaceholderText("Username")
        self.username_e.setStyleSheet('QLineEdit {font: 15pt "Calibri"; padding: 10px; background-color: white} QLineEdit:focus{border: 2px solid rgb(64, 64, 64)}')
        self.password_e = QLineEdit()
        self.password_e.setPlaceholderText("Password")
        self.password_e.setStyleSheet('QLineEdit {font: 15pt "Calibri"; padding: 10px; background-color: white} QLineEdit:focus{border: 2px solid rgb(64, 64, 64)}')
        self.password_e.setEchoMode(QLineEdit.Password)
        self.login_b = Button("LOG IN", hover_colour=LIGHT_ORANGE, hover_border=LIGHT_ORANGE)
        self.login_b.clicked.connect(lambda: self.login_clicked())
        self.signup_b = Button("SIGN UP", GREY, 'white', GREY, GREY, 'white', GREY)
        self.signup_b.clicked.connect(lambda: self.signup_clicked())
        self.layout.addRow(self.username_e)
        self.layout.addRow(self.password_e)
        self.layout.addRow(self.login_b)
        self.layout.addRow(self.signup_b)
        gbox.setLayout(self.layout)
        gbox.setStyleSheet('background-color: rgb(230, 230, 230)')
        vlayout.addWidget(self.titlebar)
        self.login_title_frame = QHBoxLayout()
        self.login_title = QLabel('WELCOME')
        self.login_title.setStyleSheet('QLabel {background: rgb(255, 140, 0); padding: 16px; color: white}')
        font = Font(30)
        self.login_title.setFont(font)
        self.login_title_frame.addWidget(self.login_title)
        self.login_title_frame.setContentsMargins(0.34*self.master.screen_width, 0.15*self.master.screen_height, 0.34*self.master.screen_width, 0)
        vlayout.addLayout(self.login_title_frame)
        lay = QHBoxLayout()
        lay.addWidget(gbox)
        lay.setContentsMargins(0.34*self.master.screen_width, 0, 0.34*self.master.screen_width, 0)
        vlayout.addLayout(lay)
        vlayout.setContentsMargins(0, 0, 0, 0)
        vlayout.addStretch(1)
        vlayout.setSpacing(0)
        self.master_container.setLayout(vlayout)
        self.master_layout.addWidget(self.master_container)
        self.setLayout(self.master_layout)
    
    def login_clicked(self):
        is_valid, self.session = self.session.dbh.check_login(self.username_e.text(), self.password_e.text(), self.session)
        if is_valid:
            self.master.show_window(1, MainScreen)
        else:
            self.error_label = QLabel('Username or password is incorrect')
            self.error_label.setStyleSheet('QLabel{background-color:'+WARM_RED+'; color: white; padding: 12px}')
            font = Font(12)
            self.error_label.setFont(font)
            self.layout.insertRow(0, self.error_label)
            self.username_e.setStyleSheet('QLineEdit {font: 15pt "Calibri"; padding: 10px; background-color: white; border: 2px solid '+WARM_RED+'}')
            self.password_e.setStyleSheet('QLineEdit {font: 15pt "Calibri"; padding: 10px; background-color: white; border: 2px solid '+WARM_RED+'}')
    
    def signup_clicked(self):
        self.master.show_window(4, SignupScreen)

    def keyPressEvent(self, event):
        if event.key() == 16777220:  # Enter key - 16777220
            self.login_clicked()
        elif event.key() == 16777237:  # Down key - 16777237
            self.password_e.setFocus()
        elif event.key() == 16777235:  # Up Key - 16777235
            self.username_e.setFocus()


class SignupScreen(QWidget):
    def __init__(self, master=None):
        super().__init__()
        self.master = master
        self.session = self.master.session
        self.initui()
    
    def initui(self):
        self.main_layout = QVBoxLayout()
        self.signup_form = QFormLayout()
        self.signup_form.setContentsMargins(0.05*self.master.screen_width, 0.05*self.master.screen_width, 0.05*self.master.screen_width, 0.05*self.master.screen_width)
        self.gbox = QGroupBox()
        self.titlebar = QLabel('FORTUNE')
        self.titlebar.setStyleSheet('QLabel {background: orange; padding: 16px; color: white}')
        font = Font(30)
        self.titlebar.setFont(font)
        self.username_e = QLineEdit()
        self.username_e.setPlaceholderText("Username")
        self.username_e.setStyleSheet('QLineEdit {font: 15pt "Calibri"; padding: 10px; background-color: white} QLineEdit:focus{border: 2px solid rgb(64, 64, 64)}')
        self.password_e = QLineEdit()
        self.password_e.setPlaceholderText("Password")
        self.password_e.setStyleSheet('QLineEdit {font: 15pt "Calibri"; padding: 10px; background-color: white} QLineEdit:focus{border: 2px solid rgb(64, 64, 64)}')
        self.password_e.setEchoMode(QLineEdit.Password)
        self.signup_b = Button("SIGN UP", hover_colour=LIGHT_ORANGE, hover_border=LIGHT_ORANGE)
        self.signup_b.clicked.connect(lambda: self.signup_clicked())
        self.signup_form.addRow(self.username_e)
        self.signup_form.addRow(self.password_e)
        self.signup_form.addRow(self.signup_b)
        self.gbox.setLayout(self.signup_form)
        self.gbox.setStyleSheet('background-color: rgb(230, 230, 230)')
        self.main_layout.addWidget(self.titlebar)
        self.lay = QHBoxLayout()
        self.lay.addWidget(self.gbox)
        self.lay.setContentsMargins(0.34*self.master.screen_width, 0.10*self.master.screen_width, 0.34*self.master.screen_width, 0)
        self.main_layout.addLayout(self.lay)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.addStretch(1)
        self.setLayout(self.main_layout)
    
    def signup_clicked(self):
        if 7 < len(self.password_e.text()) < 24:
            record = [self.username_e.text(), self.password_e.text()]
            is_valid = self.session.dbh.add_user_to_db(record)
            if is_valid:
                self.master.show_window(0, LoginScreen)
            else:
                self.error_label = QLabel("Username is taken")
                self.error_label.setStyleSheet('QLabel{background-color:'+WARM_RED+'; color: white; padding: 12px}')
                font = Font(12)
                self.error_label.setFont(font)
                self.signup_form.insertRow(0, self.error_label)
        else:
            self.error_label = QLabel('Password has to have 8-24 characters')
            self.error_label.setStyleSheet('QLabel{background-color:' + WARM_RED + '; color: white; padding: 12px}')
            font = Font(12)
            self.error_label.setFont(font)
            self.signup_form.insertRow(0, self.error_label)
            self.username_e.setStyleSheet(
                'QLineEdit {font: 15pt "Calibri"; padding: 10px; background-color: white; border: 2px solid ' + WARM_RED + '}')
            self.password_e.setStyleSheet(
                'QLineEdit {font: 15pt "Calibri"; padding: 10px; background-color: white; border: 2px solid ' + WARM_RED + '}')
        
                
class MainScreen(QWidget):
    def __init__(self, master=None):
        super().__init__()
        self.master = master
        self.session = self.master.session
        self.initui()

    def initui(self):
        self.setStyleSheet(f'background-color: {DARK_GREY}')
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        menu_bar = Menubar(self)
        self.time_buttons_gbox = QGroupBox()
        self.time_buttons_gbox.setStyleSheet('QGroupBox {background-color:'+DARK_GREY+'; border: 0px}')
        self.time_buttons_layout = QHBoxLayout()
        self.time_buttons_gbox.setLayout(self.time_buttons_layout)
        for x in ('1M', '3M', '1Y', '5Y', 'All'):
            if self.session.timeperiod == x:
                style = 'QPushButton{'
                style += f'color:{LIGHT_ORANGE}; background-color:{DARK_GREY}; border: 0px; padding: 16px'
                style += '}'
            else:
                style = 'QPushButton{'
                style += f'color:{LIGHT_GREY}; background-color:{DARK_GREY}; border: 0px; padding: 16px'
                style += '}'
                style += 'QPushButton:Hover{'
                style += f'color: white'
                style += '}'
            font = Font(12)
            time_button = QPushButton(x)
            time_button.setFont(font)
            time_button.setStyleSheet(style)
            self.time_buttons_layout.addWidget(time_button)
            time_button.clicked.connect(partial(self.view_timeframe, time=x))
        menu_bar.home_b.setStyleSheet(f'color:{LIGHT_ORANGE}; background-color:{GREY}; border: 0px; padding: 10px')
        self.main_layout.addWidget(menu_bar)
        self.main_layout.addWidget(self.time_buttons_gbox)
        self.predict_f = QHBoxLayout()
        self.predict_gbox = QGroupBox()
        self.predict_gbox.setStyleSheet('QGroupBox {background-color:'+DARK_GREY+'; border: 0px}')
        self.predict_gbox.setLayout(self.predict_f)
        self.predict_f.setContentsMargins(0.4*self.master.screen_width, 0, 0.4*self.master.screen_width, 0.05*self.master.screen_height)
        self.predict_b = Button('PREDICT', hover_colour=LIGHT_ORANGE, hover_border=LIGHT_ORANGE)
        self.predict_b.clicked.connect(lambda: self.predict_clicked())
        self.predict_f.addWidget(self.predict_b)

        # Drawing the graph.
        f = Figure(figsize=(5, 5), dpi=100)
        f.patch.set_facecolor('#202020')
        #f.patch.set_facecolor('#ffffff')
        a = f.add_subplot(111)
        canvas = FigureCanvas(f)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.main_layout.addWidget(canvas)
        self.main_layout.addWidget(self.predict_gbox)
        if self.session.current_stock:            
            if self.session.stock_change:
                self.get_stock_data()
            self.plot_graph(a)
            canvas.draw()
        self.setLayout(self.main_layout)

    def get_stock_data(self):
        self.session.stock_change = False
        df = yf.download(self.session.stock_dict[self.session.current_stock])
        self.session.current_df = df

    def view_timeframe(self, time):
        self.session.time_change = True
        self.session.timeperiod = time
        self.master.show_window(1, MainScreen)
    
    def plot_graph(self, subplot):
        df = self.session.current_df
        current_date = datetime.datetime.now()
        df = df['Close']
        if self.session.timeperiod == 'All':
            if len(df):
                df.plot(ax=subplot, lw=0.75, title=self.session.current_stock_ticker(), color='#FF8C00')
        elif self.session.timeperiod == '5Y':
            year = current_date.year-5
            df = df.loc[f'{year}-{current_date.month}-{current_date.day}':]
            if len(df):
                df.plot(ax=subplot, lw=0.75, title=self.session.current_stock_ticker(), color='#FF8C00')
        elif self.session.timeperiod == '1Y':
            year = current_date.year-1
            df = df.loc[f'{year}-{current_date.month}-{current_date.day}':]
            if len(df):
                df.plot(ax=subplot, lw=0.75, title=self.session.current_stock_ticker(), color='#FF8C00')
        elif self.session.timeperiod == '3M':
            month = (current_date.month-3) % 12
            year = current_date.year
            if month != current_date.month-3:
                year = current_date.year-1
            df = df.loc[f'{year}-{month}-{current_date.day}':]
            if len(df):
                df.plot(ax=subplot, lw=0.75, title=self.session.current_stock_ticker(), color='#FF8C00')
        elif self.session.timeperiod == '1M':
            month = (current_date.month-1) % 12
            year = current_date.year
            if month != current_date.month-1:
                year = current_date.year-1
            df = df.loc[f'{year}-{month}-{current_date.day}':]
            if len(df):
                df.plot(ax=subplot, lw=0.75, title=self.session.current_stock_ticker(), color='#FF8C00')
    
    def predict_clicked(self):
        self.pred_screen = MasterPredictionScreen(self)

class LoadingScreen(QWidget):
    def __init__(self, master=None):
        super().__init__()
        self.master = master
        self.session = self.master.session
        self.initui()
        self.thread = BackgroundTask(self)
        self.thread.completed.connect(self.complete)
        self.thread.start()
        self.show()
    
    def initui(self):
        self.main_layout = QHBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.loading_screen = QWidget()
        
        # Loading screen.
        self.loading_gbox = QGroupBox()
        self.loading_gbox.setStyleSheet(f'QGroupBox {{background-color: {DARK_GREY}; border: 0px}}')
        self.loading_layout = QVBoxLayout()
        self.loading_layout.setContentsMargins(0, 0, 0, 0)
        self.loading_gbox.setLayout(self.loading_layout)
        self.loading_label = QLabel("GENERATING PREDICTION ...")
        self.loading_label.setStyleSheet('QLabel {color: white}')
        self.loading_label.setFont(Font(20))
        self.loading_layout.addWidget(self.loading_label)
        self.loading_container = QHBoxLayout()
        self.loading_container.setContentsMargins(0, 0, 0, 0)
        self.loading_container.addWidget(self.loading_gbox)
        self.loading_screen.setLayout(self.loading_container)
        self.main_layout.addWidget(self.loading_screen)
        self.setLayout(self.main_layout)
    
    def complete(self, prediction):
        self.master.show_window(1, PredictionScreen, prediction)


class PredictionScreen(QWidget):
    def __init__(self, prediction, session):
        super().__init__()
        self.predictions = prediction
        print(self.predictions)
        self.session = session
        if len(self.predictions) != 2:
            self.initui()
    
    def initui(self):
        self.container = QWidget()
        self.container.setStyleSheet(f'QWidget{{background-color: {DARK_GREY}}}')
        self.ticker_layout = QHBoxLayout()
        self.ticker_l = Label(self.session.current_stock_ticker())
        self.ticker_layout.addWidget(self.ticker_l)
        self.ticker_layout.addStretch()
        self.main_layout = QVBoxLayout()
        self.labels_l = QHBoxLayout()
        self.prediction_head_l = QLabel('PREDICTION')
        self.prediction_head_l.setStyleSheet(f'QLabel{{color: {LIGHT_ORANGE}; background-color: {GREY}; padding: 16px}}')
        self.prediction_head_l.setFont(Font(12))
        self.prediction_head_l.setAlignment(Qt.AlignCenter)
        self.accuracy_head_l = QLabel('ACCURACY')
        self.accuracy_head_l.setStyleSheet(f'QLabel{{color: {LIGHT_ORANGE}; background-color: {GREY}; padding: 16px}}')
        self.accuracy_head_l.setFont(Font(12))
        self.accuracy_head_l.setAlignment(Qt.AlignCenter)
        self.days_head_l = QLabel('DAYS')
        self.days_head_l.setStyleSheet(f'QLabel{{color: {LIGHT_ORANGE}; background-color: {GREY}; padding: 16px}}')
        self.days_head_l.setFont(Font(12))
        self.days_head_l.setAlignment(Qt.AlignCenter)
        self.labels_l.addWidget(self.prediction_head_l)
        self.labels_l.addWidget(self.accuracy_head_l)
        self.labels_l.addWidget(self.days_head_l)
        self.labels_l.setSpacing(1)
        self.labels_l.setContentsMargins(0, 0, 0, 0)
        self.main_layout.addLayout(self.ticker_layout)
        self.main_layout.addLayout(self.labels_l)
        self.container.setLayout(self.main_layout)
        self.con_lay = QHBoxLayout()
        self.con_lay.addWidget(self.container)
        self.con_lay.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.con_lay)
        days_list = [1, 3, 7]
        for i in range(3):
            prediction = self.predictions[i]
            prediction_layout = QHBoxLayout()

            if prediction[0] == 1:
                image = QPixmap('images/up_arrow.png')
            else:
                image = QPixmap('images/down_arrow.png')
            
            self.prediction_l = QLabel()
            self.prediction_l.setPixmap(image)
            self.prediction_l.setAlignment(Qt.AlignCenter)
            self.accuracy_l = QLabel(f'{str(prediction[1])}%')
            self.accuracy_l.setStyleSheet(f'QLabel{{color: white}}')
            self.accuracy_l.setFont(Font(20))
            self.accuracy_l.setAlignment(Qt.AlignCenter)
            self.days_l = QLabel(f'{str(days_list[i])}')
            self.days_l.setStyleSheet(f'QLabel{{color: white}}')
            self.days_l.setFont(Font(20))
            self.days_l.setAlignment(Qt.AlignCenter)
            prediction_layout.addWidget(self.prediction_l)
            prediction_layout.addWidget(self.accuracy_l)
            prediction_layout.addWidget(self.days_l)
            self.main_layout.addLayout(prediction_layout)
        # self.save_button = Button('SAVE')
        # self.main_layout.addWidget(self.save_button)
        self.setLayout(self.main_layout)


class MasterPredictionScreen(QDialog):
    def __init__(self, master):
        super().__init__(master)
        self.setModal(True)
        self.master = master
        self.session = master.session
        self.setWindowTitle('Fortune')
        self.initui()
    
    def initui(self):
        self.stack = QStackedWidget(self)
        self.stack.addWidget(LoadingScreen(self))
        self.stack.addWidget(PredictionScreen(['', ''], self.session))
        hbox = QHBoxLayout(self)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(self.stack)
        self.setLayout(hbox)
        self.stack.setCurrentIndex(0)
        self.show()
    
    def show_window(self, i, widget, a):
        self.stack.removeWidget(self.stack.widget(i))
        self.stack.insertWidget(i, widget(a, self.session))
        self.stack.setCurrentIndex(i)


class ViewMyStocksScreen(QWidget):
    def __init__(self, master=None):
        super().__init__()
        self.master = master
        self.session = self.master.session
        self.initui()
    
    def initui(self):
        self.stock_groupbox = QWidget()
        g_style = 'QWidget{'
        g_style += f'border: 1px solid {DARK_GREY}; background-color: {DARK_GREY}'
        g_style += '}'
        self.stock_groupbox.setStyleSheet(g_style)
        self.stock_layout = QFormLayout()
        self.buttons = []
        for stock_id in self.session.stock_dict:
            frame = QHBoxLayout()
            stock_ticker = self.session.stock_dict[stock_id]
            view_b = Button(stock_ticker, hover_colour=LIGHT_ORANGE, hover_border=LIGHT_ORANGE)
            view_b.clicked.connect(partial(self.view_func, stock=stock_id))
            delete_b = Button("DELETE", GREY, 'white', GREY, GREY, 'white', GREY)
            delete_b.clicked.connect(partial(self.delete_func, stock=stock_id))
            frame.addWidget(view_b)
            frame.addStretch()
            frame.addWidget(delete_b)
            self.stock_layout.addRow(frame)
        self.stock_groupbox.setLayout(self.stock_layout)
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.stock_groupbox)
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet('QScrollArea{border: 0px}')
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        self.menubar = Menubar(self)
        self.menubar.view_b.setStyleSheet(f'color:{LIGHT_ORANGE}; background-color:{GREY}; border: 0px; padding: 16px')
        self.main_layout.addWidget(self.menubar)
        self.main_layout.addWidget(self.scroll)
    
    def view_func(self, stock):
        self.session.current_stock = stock  # Changes the current stock to the chosen stock.
        self.session.stock_change = True  # Downloads a new dataframe.
        self.master.show_window(1, MainScreen)  # Goes back to the home screen.
    
    def delete_func(self, stock):
        pass
    
class SearchScreen(QWidget):
    def __init__(self, master=None):
        super().__init__()
        self.master = master
        self.session = self.master.session
        self.initui()
    
    def initui(self):
        self.widget_list = []
        self.master_layout = QHBoxLayout()
        self.setStyleSheet(f'background-color: {DARK_GREY}')
        self.master_container = QWidget()
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        self.menubar = Menubar(self)
        self.menubar.addstock_b.setStyleSheet(f'color:{LIGHT_ORANGE}; background-color:{GREY}; border: 0px; padding: 16px')
        self.search_bar = QHBoxLayout()
        self.search_bar.setContentsMargins(20, 20, 20, 20)
        self.search_bar.setSpacing(10)
        self.search_e = QLineEdit()
        self.search_e.setPlaceholderText("Enter Stock Ticker")
        self.search_e.setStyleSheet('QLineEdit {font: 15pt "Calibri"; padding: 10px; background-color: white} QLineEdit:focus{border: 2px solid rgb(64, 64, 64)}')
        self.search_b = Button("  GO  ", hover_colour=LIGHT_ORANGE, hover_border=LIGHT_ORANGE)
        self.search_b.clicked.connect(lambda: self.search_func())
        self.search_bar.addWidget(self.search_e)
        self.search_bar.addWidget(self.search_b)
        self.main_layout.addWidget(self.menubar)
        self.main_layout.addLayout(self.search_bar)
        self.error_label = QLabel("Stock is invalid")
        self.error_label.setStyleSheet('QLabel{background-color:'+WARM_RED+'; color: white; padding: 12px}')
        font = Font(12)
        self.error_label.setFont(font)
        self.error_frame = QHBoxLayout()
        self.error_frame.setContentsMargins(0, 0, 0.8*self.master.screen_width, 0)
        self.error_frame.addWidget(self.error_label)
        self.main_layout.insertLayout(1, self.error_frame)
        self.error_label.setHidden(True)
        self.f = Figure(figsize=(5, 5), dpi=100)
        self.f.patch.set_facecolor('#202020')
        self.a = self.f.add_subplot(111)
        self.canvas = FigureCanvas(self.f)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.main_layout.addWidget(self.canvas)
        self.add_f = QHBoxLayout()
        self.add_f.setContentsMargins(0.4*self.master.screen_width, 0, 0.4*self.master.screen_width, 0.1*self.master.screen_height)
        self.add_b = Button("ADD STOCK", hover_colour=LIGHT_ORANGE, hover_border=LIGHT_ORANGE)
        self.add_b.clicked.connect(lambda: self.add_clicked())
        self.add_f.addWidget(self.add_b)
        self.main_layout.addLayout(self.add_f)
        self.master_container.setLayout(self.main_layout)
        self.master_layout.addWidget(self.master_container)
        self.master_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.master_layout)
    
    def search_func(self):
        is_valid = False
        stock_ticker = str(self.search_e.text()).upper()
        try:
            df = yf.download(stock_ticker)
        except ValueError:
            df = []
        if len(df):  # If the stock ticker is valid.
            is_valid = True        
            self.error_label.setHidden(True)
            self.search_e.setStyleSheet('QLineEdit {font: 15pt "Calibri"; padding: 10px; background-color: white} QLineEdit:focus{border: 2px solid rgb(64, 64, 64)}')
            df = df['Close']
            df.plot(ax=self.a, lw=0.75, title=stock_ticker, color='#FF8C00')              
            self.canvas.draw()
            
        if not is_valid:
            self.error_label.setHidden(False)
            self.search_e.setStyleSheet('QLineEdit {font: 15pt "Calibri"; padding: 10px; background-color: white; border: 2px solid '+WARM_RED+'}')
    
    def add_clicked(self):
        stock = str(self.search_e.text()).upper()
        is_valid = self.session.dbh.add_stock_to_db(stock, self.session.current_user.user_id)
        if is_valid:
            self.search_e.setStyleSheet('QLineEdit {font: 15pt "Calibri"; padding: 10px; background-color: white} QLineEdit:focus{border: 2px solid rgb(64, 64, 64)}')
            self.session.update()
            self.master.show_window(2, ViewMyStocksScreen)
        else:
            self.error_label.setHidden(False)
            self.search_e.setStyleSheet('QLineEdit {font: 15pt "Calibri"; padding: 10px; background-color: white; border: 2px solid '+WARM_RED+'}')

    def keyPressEvent(self, event):
        if event.key() == 16777220:  # Enter key - 16777220
            self.search_func()

class Menubar(QGroupBox):
    def __init__(self, parent_layout):
        super().__init__()
        font = Font(12)
        self.style = 'QPushButton{'
        self.style += f'color:{LIGHT_GREY}; background-color:{GREY}; border: 0px; padding: 16px'
        self.style += '}'
        self.style += 'QPushButton:Hover{'
        self.style += f'color: white'
        self.style += '}'
        self.setStyleSheet('QGroupBox {background-color: '+GREY+'; border: 0px}')
        self.parent_layout = parent_layout
        self.layout = QHBoxLayout()
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.home_b = QPushButton('HOME')
        self.home_b.clicked.connect(lambda: self.parent_layout.master.show_window(1, MainScreen))
        self.home_b.setFont(font)
        self.home_b.setStyleSheet(self.style)
        self.home_b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.layout.addWidget(self.home_b)
        self.view_b = QPushButton('MY STOCKS')
        self.view_b.clicked.connect(lambda: self.parent_layout.master.show_window(2, ViewMyStocksScreen))
        self.view_b.setFont(font)
        self.view_b.setStyleSheet(self.style)
        self.view_b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.layout.addWidget(self.view_b)
        self.addstock_b = QPushButton('SEARCH')
        self.addstock_b.clicked.connect(lambda: self.parent_layout.master.show_window(3, SearchScreen))
        self.addstock_b.setFont(font)
        self.addstock_b.setStyleSheet(self.style)
        self.addstock_b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.layout.addWidget(self.addstock_b)
        self.logout_b = QPushButton('LOG OUT')
        self.logout_b.clicked.connect(lambda: self.parent_layout.master.show_window(0, LoginScreen))
        self.logout_b.setFont(font)
        self.logout_b.setStyleSheet(self.style)
        self.layout.addWidget(self.logout_b)
        self.setLayout(self.layout)


class BackgroundTask(QThread):
    completed = pyqtSignal(list)

    def __init__(self, master):
        super().__init__()
        self.session = master.session

    def run(self):
        predictions = []
        stock = self.session.current_stock_ticker()
        for period in (1, 3, 7):          
            dnn = Dnn(stock=stock, period=period)
            prediction = dnn.predict()
            predictions.append(prediction)

        self.completed.emit(predictions)


class Button(QPushButton):
    def __init__(self, text, border_colour=ORANGE, bg=ORANGE, fg='white', hover_colour=None, hover_font=None, hover_border=None):
        super().__init__(text)
        self.style = f'QPushButton {{' \
                     f'color:{fg}; border-width:2px; border-style:solid; border-color:{border_colour};' \
                     f'border-radius:24px; padding:16px; background-color:{bg}}}' \
                     f'QPushButton:Hover {{' \
                     f'background-color: {hover_colour}; color: {hover_font}; border-color: {hover_border}}}'
        self.setStyleSheet(self.style)
        font = Font(12)
        self.setFont(font)


class Label(QLabel):
    def __init__(self, text):
        super().__init__(text)
        self.style = f'QLabel {{' \
                     f'color:white; border-width:2px; border-style:solid; border-color:{ORANGE}; border-radius:24px;' \
                     f'padding:16px; background-color:{ORANGE}}}'
        self.setStyleSheet(self.style)
        font = Font(12)
        self.setFont(font)   


class Font(QFont):
    def __init__(self, size):
        super().__init__()
        self.setLetterSpacing(QFont.AbsoluteSpacing, 2)
        self.setWeight(QFont.ExtraBold)
        self.setStretch(110)
        self.setPixelSize(size)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    sizeObject = QDesktopWidget().screenGeometry(-1)
    WIDTH = sizeObject.width()
    HEIGHT = sizeObject.height()
    app.setWindowIcon(QIcon('logo.png'))
    ex = MainUi()
    sys.exit(app.exec())
