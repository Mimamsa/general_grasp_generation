import time
from log import Log


class Timer:
    """ Use for measure time. Initail then using by start and end function.
    """
    def __init__(self, name, showEach=False, rate=50):
        """
        Args:
            name (str): The title of this time.
            showEach (bool): Record each time for debug or not.
            rate (int): The average times.
        """
        self.__name = name
        self.__initial = 0.0
        self.__time = 0.0
        self.__average = 0.0
        self.__count = 0
        self.__rate = rate
        self.__showEach = showEach

    def start(self):
        self.__initial = time.time()

    def end(self, deno=1):
        """
        Args:
            deno (int): If execute same step more than one times. use deno for average.
        Example:
            If you need to measure a function that process three data.
            >>> start()
            >>> processThressData([10,5,1])
            >>> end(3)
        """
        self.__time = time.time() - self.__initial
        self.__average = self.__average + (self.__time / deno)

        if self.__showEach == True:
            Log.debug(self.__name + " time: " + str(self.__time)[0:6] + " sec.")
        
        self.__count += 1
        if self.__count == self.__rate:
            self.__average = self.__average / self.__rate
            avg = '{0:.5f}'.format(self.__average)
            Log.info("==== Average " + self.__name + " time(" + str(self.__rate) + " times): " + avg + " ====", "blue")
            #Log.addKeepMsg("Average " + self.__name + " time", avg)
            self.__average = 0
            self.__count = 0
        
        return self.__time

