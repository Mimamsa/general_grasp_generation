""" Set the log format, color and type.
Auther: Jason
Date: 2019/02/12
"""
import os, datetime, logging, inspect, curses, time
from curses import wrapper
from threading import Thread
from termcolor import colored
from collections import OrderedDict
from logging.handlers import TimedRotatingFileHandler

class Log():
    __path = './logs/'
    if not os.path.isdir(__path):
        os.makedirs(__path)
    __level = 'INFO'
    __dashboard = False
    __keepShow = False
    __systemDict = OrderedDict()
    __systemDict["systemStatus"] = "======== System Status ========"

    def __getMode(mode):
        return {
            'CRITICAL': logging.CRITICAL,
            'ERROR': logging.ERROR,
            'WARNING': logging.WARNING,
            'INFO': logging.INFO,
            'DEBUG': logging.DEBUG
        }.get(mode, logging.NOTSET)

    def __getScope():
        """ Get the function call scope.
        Returns:
            filename: if file1.py call getScope, it will return "file1".
        """
        frame = inspect.stack()[2]
        module = inspect.getmodule(frame[0])
        filename = module.__file__
        if filename.rfind("/") >= 0:
            filename = filename[filename.rfind("/")+1:]
        filename = filename.replace(".py","")
        return filename
    

    # def __show(stdscr):
    #     curses.use_default_colors()
    #     while Log.__keepShow:
    #         stdscr.clear()
    #         i=0
    #         for key, value in Log.__systemDict.items():
    #             stdscr.addstr(i, 0, value)
    #             i+=1
    #         stdscr.refresh()
    #         time.sleep(0.5)

    def reload():
        """ Reload the logging with configuration.
        """
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logFilename = datetime.datetime.now(). strftime(Log.__path + "log.txt")
        fileHandler = TimedRotatingFileHandler(logFilename, when="midnight", interval=1)
        fileHandler.suffix = "%Y%m%d"
        stringFormat = colored('[%(asctime)s.%(msecs)03d]', 'green') + \
            colored(' %(name)-10s', 'blue') + \
            colored(' %(levelname)-8s', attrs=['dark', 'bold']) + \
            '%(message)s'

        logging.basicConfig(level = Log.__getMode(Log.__level),
            format = stringFormat,
            handlers=[
            fileHandler,
            logging.StreamHandler()
            ])
        #Log.info("Set log - path:{}, level:{}, dashboard:{}".format(Log.__path, Log.__level, Log.__dashboard))

    def set(path=None, level=None, dashboard=None):
        """ Reset the program's mode
        Args:
            path (str): the log path.
            level (str): logger level.
            dashboard (bool): on/off dashboard mode.
        Example:
            Set mode when initial your program.
            This will set logger for DEBUG mode and open the dashboard.
            >>> setMode("../whereToSave/","DEBUG", True)
        """
        if path and os.path.isdir(path):
            if not os.path.isdir(path + "/logs/"):
                os.makedirs(path + "/logs/")
                #os.system("mv ../logs/ " + str(path))
            Log.__path = path + "/logs/"
        if level:
            Log.__level = level
        if dashboard:
            Log.__dashboard = dashboard
        Log.reload()

    """ Save log by each logger mode.
    Args:
        msg (str): message to save.
        color (termcolor.color): Display by which color.
        attrs (list): Attributes for text.
    """
    def debug(msg, color="cyan", attrs=['bold']):
        logger = logging.getLogger(Log.__getScope())
        logger.debug(colored(str(msg), color, attrs = attrs))

    def info(msg, color="green", attrs=['bold']):
        logger = logging.getLogger(Log.__getScope())
        logger.info(colored(str(msg), color, attrs = attrs))

    def warning(msg, color="yellow", attrs=None):
        logger = logging.getLogger(Log.__getScope())
        logger.warning(colored(str(msg), color, attrs = attrs))

    def error(msg, color="red", attrs=None):
        logger = logging.getLogger(Log.__getScope())
        logger.error(colored(str(msg), color, attrs = attrs))
    
    def critical(msg, color="red", attrs=['bold']):
        logger = logging.getLogger(Log.__getScope())
        logger.critical(colored(str(msg), color, attrs = attrs))

    # def addKeepMsg(topic, data):
    #     """ Put the information what you want to keep in dashborad.
    #     Args:
    #         topic: The title of your data.
    #         data: The information of data.
    #     Example:
    #         This will create a time information on your dashborad 1 by Log.show().
    #         >>> addKeepMsg("The time", time.time())
    #     """
    #     if not Log.__dashboard:
    #         return
    #     output = str(topic) + ": " + str(data)
    #     Log.__systemDict[topic] = output

    #     if Log.__keepShow == False:
    #         Log.__keepShow = True
    #         processThread = Thread(target=wrapper,args=(Log.__show,)) 
    #         processThread.start()

    # def closeDashboard():
    #     Log.__keepShow = False
    #     time.sleep(0.3)
    
Log.reload()
