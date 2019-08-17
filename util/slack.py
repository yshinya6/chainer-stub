#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests, sys, os
sys.path.append(os.path.dirname(__file__))
from asset import *
# Please make asset.py contatining TOKEN, ICON, NAME, and PROXIES.


class SlackWrapper:
    #slack
    __token = TOKEN
    __channel = '#yourchannel' 
    __postSlackUrl = 'https://slack.com/api/chat.postMessage'
    __icon_url = ICON
    __username = NAME

    def __init__(self):
        pass

    def format_summary(self, data:dict):
        pattern = data['summary']['pattern']
        mean = data['summary']['mean_accuracy']
        std = data['summary']['std_accuracy']
        message = """ 
        *Experiment Report*
        *Pattern*: {}
        *Mean Acc.*: {}
        *Std*: {}
        """.format(pattern, mean, std)
        return message


    def report_fail(self, msg:str):
        message = '*[Failed]* ' + msg
        self.post(message)

    def report_summary(self, data:dict):
        message = self.format_summary(data)
        self.post(message)

    def post(self, posttext):
        params = {'token': self.__token, 
                  'channel': self.__channel , 
                  'text':posttext,
                  'icon_emoji': self.__icon_url,
                  'username':self.__username,
                  'unfurl_links': 'false'
                  }
        requests.post(self.__postSlackUrl, params=params, proxies=PROXIES)  

