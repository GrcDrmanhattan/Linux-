#include "server.hpp"
#include <iostream>

using namespace std;

Server::Server()
{
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port=htons(SERVER_PORT);
    serverAddr.sin_addr.s_addr=inet_addr(SERVER_IP);

    listener = 0;
    epfd = 0;
    
}

void Server::Init()
{
    cout<<"初始化服务器..."<<endl;

    listener = socket(AF_INET,SOCK_STREAM,0);
    if(listener<0)
    {
        perror("scoket error\n");
        exit(-1);
    }

    if(bind(listener,(sockaddr*)&serverAddr,sizeof(serverAddr)) <0)
    {
        perror("bind error\n");
        exit(-1);
    }

    int ret = listen(listener,5);
    if(ret<0)
    {
        perror("listen error\n");
        exit(-1);
    }


    cout<<"开始listen: "<<SERVER_IP<<endl;


    epfd=epoll_create(EPOLL_SIZE);
    if(epfd<0)
    {
        perror("epoll create error");
        exit(-1);
    }

    addfd(epfd,listener,true);

}

void Server::Close()
{
    ::close(epfd);
    ::close(listener);
}

 Server::  ~Server()
 {
     Close();
 }

int Server::send2AllMsg(int clientfd)
{
    //buf rec new msg
    //message save format message
   char buf[BUF_SIZE];
   char message[BUF_SIZE];
   bzero(buf,BUF_SIZE);
   bzero(message,BUF_SIZE);

   cout<<"read from client: clientid="<< clientfd<<endl;
   int len = recv(clientfd,buf,BUF_SIZE,0);

   //如果客户关闭连接
   if(len==0)
   {
       ::close(clientfd);
       //Close();//这样会一个客户退出，导致服务器和其他的客户也退出
       clients_list.remove(clientfd);

       cout<<"clientId: "<<clientfd<<" closed,还有 "<<
           clients_list.size()<<"个人在聊天室"<<endl;
   }
   //发送广播信息
   else
   {
       if(clients_list.size()==1)
       {
           send(clientfd,CAUTION,strlen(CAUTION),0);
           return len;
       }
       //将格式化字符输出到message
       sprintf(message,SERVER_MSG,clientfd,buf);

       list<int>::iterator it;
        for(it = clients_list.begin();it != clients_list.end();++it)
        {
            if(*it != clientfd)
            {
                if(send(*it,message,BUF_SIZE,0)<0)
                {
                    return -1;
                }
            }
        }

   }

   return len;
}

void Server::Start()
{
    //static
    //
    struct epoll_event events[EPOLL_SIZE];
    Init();

    while(1)
    {
        int epoll_events_cont = epoll_wait(epfd,events,EPOLL_SIZE,-1);
        
        if(epoll_events_cont < 0)
        {
            perror("epoll error");
            break;
        }

        cout<<"epoll_events_cont =\n"
            <<epoll_events_cont<<endl;

        for(int i=0;i<epoll_events_cont;++i)
        {
            int sockfd = events[i].data.fd;
            //前面已经把listener监听了放入epfd中
            //新客户连接
            //如果等于listener代表服务器有事件发生
            if(sockfd == listener)
            {
                sockaddr_in client_addr;
                socklen_t client_adr_len = sizeof(client_addr);
                int clientfd = accept(listener,(sockaddr*)&client_addr,&client_adr_len);

                cout<<"client connection from: "
                    <<inet_ntoa(client_addr.sin_addr)<<" : "
                    <<ntohs(client_addr.sin_port)<<", clientfd = "
                    <<clientfd<<endl;

                addfd(epfd,clientfd,true);

                //
                //
                clients_list.push_back(clientfd);
                cout<<"add new client = "<<clientfd<<"to epoll "
                    <<endl;

                cout<<"现在有"<<clients_list.size()<<"个人在聊天室 "<<endl;


                cout<<"welcome message"<<endl;
                char message[BUF_SIZE];
                memset(&message,0,BUF_SIZE);


                sprintf(message,SERVER_WELCOME ,clientfd);
                int ret = send(clientfd,message,BUF_SIZE,0);
                if(ret<0)
                {
                    perror("send error");
                    Close();
                    exit(-1);
                }
            }
            //客户端有事件发生
            //处理从客户端的消息，广播消息
            else
            {
                int ret = send2AllMsg(sockfd);
                if(ret <0)
                {
                    perror("");
                    Close();
                    exit(-1);
                }
            }

        }

    }

    Close();
}


