#pragma once

#include <vector>

#include <Windows.h>

// #include <exception.h>




class StopWatchInterface {
public:
  StopWatchInterface() {}
  virtual ~StopWatchInterface() {}

public:

  virtual void start() = 0;
  virtual void stop() = 0;

  //Reset time counters to zero
  virtual void reset() = 0;

  //! Time in msec. after start. If the stop watch is still running (i.e. there
  //! was no call to stop()) then the elapsed time is returned, otherwise the
  //! time between the last start() and stop call is returned
  virtual float getTime() = 0;

  //! Mean time to date based on the number of times the stopwatch has been
  //! _stopped_ (ie finished sessions) and the current total time
  virtual float getAverageTime() = 0;
};


class StopWatchWin : public StopWatchInterface{
public:
  StopWatchWin()
    : start_time(),
      end_time(),
      diff_time(0.0f),
      total_time(0.0f),
      isRunning(false),
      clock_sessions(0),
      freq(0),
      freq_set(false) {
        // 调用系统函数，求得频率
        if (!freq_set) {
          LARGE_INTEGER temp;

          QueryPerformanceFrequency(reinterpret_cast<LARGE_INTEGER*>(&temp));

          freq = (static_cast<double>(temp.QuadPart)) / 1000.0;

          freq_set = true;

        }
      }


  ~StopWatchWin() {}


public:
  inline void start();

  inline void stop();

  //! Reset time counters to zero
  inline void reset();


  inline float getTime();

  inline float getAverageTime();



private:
  LARGE_INTEGER start_time;
  LARGE_INTEGER end_time;

  //last start和stop之间的时间差
  float diff_time;

  //所有starts和ends之间的时间差之和
  float total_time;
  bool isRunning;

  //记录多少个时间段，
  int clock_sessions;

  // 时钟频率
  double freq;

  //
  bool freq_set;
};

inline void StopWatchWin::start() {
  QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*> (&start_time));
  isRunning = true;
}


inline void StopWatchWin::stop() {
  QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*> (&end_time));
  diff_time = static_cast<float>(((static_cast<double>(end_time.QuadPart)-
                                    static_cast<double>(start_time.QuadPart))/
                                    freq));
  total_time += diff_time;
  ++clock_sessions;
  isRunning=false;
}


//重置
//不会改变isRunning的状态
inline void StopWatchWin::reset(){
  diff_time=0;
  total_time=0;
  clock_sessions=0;

  if (isRunning){
    QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&start_time));
  }
}

inline float StopWatchWin::getTime() {
  float retval = total_time;

  if (isRunning){
    LARGE_INTEGER temp;
    QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&temp));
    retval += static_cast<float>(((static_cast<double>(temp.QuadPart)-
                                    static_cast<double>(start_time.QuadPart)) / 
                                    freq));
  }
  return retval;
}

inline float StopWatchWin::getAverageTime() {
  return (clock_sessions>0)? (total_time/clock_sessions) : 0.0f;
}



////////////////////////////////////////////////////////////////////////////////
//! Timer functionality exported

//creating timer
inline bool sdkCreateTimer(StopWatchInterface **timer_interface) {
  *timer_interface = reinterpret_cast<StopWatchInterface *>(new StopWatchWin());

  return (*timer_interface != nullptr) ? true : false;
}

inline bool sdkDeleteTimer(StopWatchInterface **timer_interface) {
  if (*timer_interface) {
    delete *timer_interface;
    *timer_interface = nullptr;
  }

  return true;
}

inline bool sdkStartTimer(StopWatchInterface **timer_interface) {
  if (*timer_interface) {
    (*timer_interface)->start();
  }

  return true;
}

inline bool sdkStopTimer(StopWatchInterface **timer_interface) {
  if (*timer_interface) {
    (*timer_interface)->stop();
  }

  return true;
}

inline bool sdkResetTimer(StopWatchInterface **timer_interface) {
  if (*timer_interface) {
    (*timer_interface)->reset();
  }

  return true;
}


inline float sdkGetAverageTimerValue(StopWatchInterface **timer_interface) {
  if (*timer_interface) {
    return (*timer_interface)->getAverageTime();
  } else {
    return 0.0f;
  }
}

inline float sdkGetTimerValue(StopWatchInterface **timer_interface) {
  if (*timer_interface) {
    return (*timer_interface)->getTime();
  } else {
    return 0.0f;
  }
}