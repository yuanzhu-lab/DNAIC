#!/usr/bin/env python

# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import subprocess
import threading
import time

# Assumes meteor-1.5.jar is in the same directory as meteor.py.  Change as needed.
METEOR_JAR = 'meteor-1.5.jar'
# print METEOR_JAR

class Meteor:

    def __init__(self):
        self.env = os.environ.copy()
        self.env['LC_ALL'] = 'en_US.UTF_8'
        self.env['JAVA_OPTS'] = '-Xmx8G'  # 增加Java堆内存
        meteor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), METEOR_JAR)
        if not os.path.exists(meteor_path):
            raise FileNotFoundError(f"meteor-1.5.jar not found at {meteor_path}")
            
        self.meteor_cmd = ['java', '-jar', '-Xmx8G', METEOR_JAR, \
                '-', '-', '-stdio', '-l', 'en', '-norm']
        try:
            self.meteor_p = subprocess.Popen(self.meteor_cmd, \
                    cwd=os.path.dirname(os.path.abspath(__file__)), \
                    stdin=subprocess.PIPE, \
                    stdout=subprocess.PIPE, \
                    stderr=subprocess.PIPE,
                    env=self.env, universal_newlines=True, bufsize=1)
            
            # 等待进程启动
            time.sleep(1)
            if self.meteor_p.poll() is not None:
                stderr_content = self.meteor_p.stderr.read()
                print(stderr_content)
                raise RuntimeError("METEOR process failed to start")
                
        except Exception as e:
            raise RuntimeError(f"Failed to start METEOR process: {str(e)}")
            
        # Used to guarantee thread safety
        self.lock = threading.Lock()
        self._process_alive = True

    def compute_score(self, gts, res):
        if not self._process_alive:
            self._restart_process()
            
        assert(gts.keys() == res.keys())
        imgIds = sorted(list(gts.keys()))
        scores = []

        eval_line = 'EVAL'
        try:
            self.lock.acquire()
            for i in imgIds:
                assert(len(res[i]) == 1)
                stat = self._stat(res[i][0], gts[i])
                eval_line += ' ||| {}'.format(stat)

            # Send to METEOR
            self._write_line(eval_line + '\n')
            
            # Collect segment scores
            for i in range(len(imgIds)):
                score = float(self._read_line(timeout=180).strip())  # 增加超时时间到180秒
                scores.append(score)

            # Final score
            final_score = float(self._read_line(timeout=180).strip())  # 增加超时时间到180秒
            
        except Exception as e:
            self._process_alive = False
            self._restart_process()  # 尝试重启进程
            raise RuntimeError(f"Error during METEOR scoring: {str(e)}")
        finally:
            self.lock.release()

        return final_score, scores

    def _write_line(self, line):
        try:
            self.meteor_p.stdin.write(line)
            self.meteor_p.stdin.flush()
        except BrokenPipeError:
            self._process_alive = False
            self._restart_process()  # 尝试重启进程
            raise

    def _read_line(self, timeout=180):  # 默认超时时间改为180秒
        start_time = time.time()
        while True:
            if self.meteor_p.poll() is not None:
                self._process_alive = False
                raise RuntimeError("METEOR process has terminated unexpectedly")
                
            if self.meteor_p.stdout.readable():
                line = self.meteor_p.stdout.readline()
                if line:
                    return line
            if time.time() - start_time > timeout:
                self._process_alive = False
                raise TimeoutError(f"METEOR process response timeout after {timeout} seconds")
            time.sleep(0.1)

    def _restart_process(self):
        self.__del__()
        self.__init__()

    def method(self):
        return "METEOR"

    def _stat(self, hypothesis_str, reference_list):
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        try:
            self._write_line(score_line + '\n')
            return self._read_line(timeout=180).strip()  # 增加超时时间到180秒
        except Exception as e:
            self._process_alive = False
            self._restart_process()  # 尝试重启进程
            raise RuntimeError(f"Error in _stat: {str(e)}")

    def __del__(self):
        try:
            self.lock.acquire()
            if hasattr(self, 'meteor_p'):
                self.meteor_p.stdin.close()
                self.meteor_p.terminate()
                try:
                    self.meteor_p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.meteor_p.kill()
                    self.meteor_p.wait()
        except Exception:
            pass
        finally:
            if hasattr(self, 'lock'):
                self.lock.release()
