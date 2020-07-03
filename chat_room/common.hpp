#ifndef CHARTROOM_COMMON_H
#define CHARTROOM_COMMON_H

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <iostream>
#include <list>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/epoll.h>
#include <sys/socket.h>
#include <unistd.h>
#include <pthread.h>

using namespace std;

//默认ip
#define SERVER_IP "127.0.0.1"
//端口
#define SERVER_PORT 7887
//epoll 最大处理数目
#define EPOLL_SIZE 5000
//buffer size 65536
#define BUF_SIZE 0xffff
//欢迎信息
#define SERVER_WELCOME "欢迎来到聊天室，客户id ：is##%d"
//
#define SERVER_MSG "CLientID %d say >> %s"
//退出room、
#define EXIT "EXIT"
//提示信息只有一个人
#define CAUTION "只有你一个在room"
//添加文件描述符到epollfd
//enable true默认et边沿触发，false：LT水平触发
static void addfd(int epollfd,int fd,bool enable_et)
{
    epoll_event ev;
    ev.events=EPOLLIN;//epoll事件
    ev.data.fd=fd;
    
    if(enable_et)
    {
        ev.events = EPOLLIN | EPOLLET;
        
    }
    epoll_ctl(epollfd,EPOLL_CTL_ADD,fd,&ev);
    
    int  flag = fcntl(fd,F_GETFL);
    flag |= O_NONBLOCK;
    fcntl(fd,F_SETFL,flag);
    
    cout<<"fd added to epoll"<<endl;
    
}



#endif

