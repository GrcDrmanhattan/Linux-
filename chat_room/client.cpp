#include <iostream>

#include "client.hpp"

using namespace std;

Client::Client()
{
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(SERVER_PORT);
    serverAddr.sin_addr.s_addr = inet_addr(SERVER_IP);

    sock = 0;
    pid= 0;
    isClientWork = true;
    epfd = 0;
}

 Client::~Client()
 {
     Close();

 }

void Client::Connect()
{
    cout<< "connect server: "<<SERVER_IP<<" : "<<SERVER_PORT<<endl;

    sock = socket(AF_INET,SOCK_STREAM,0);
    if(sock<0)
    {
        perror("socket error");
        exit(-1);
    }

    if(connect(sock,(sockaddr*)&serverAddr,sizeof(serverAddr))<0)
    {
        perror("connect error");
        exit(-1);
    }

    if(pipe(pipe_fd) <0)
    {
        perror("pipe error");
        exit(-1);
    }

    epfd = epoll_create(EPOLL_SIZE);
    if(epfd<0)
    {
        perror("epfd error");
        exit(-1);
    }

    //添加sock和父进程读的管道fd0 到epfd红黑树
    addfd(epfd,sock,true);
    addfd(epfd,pipe_fd[0],true);
}

void Client::Close()
{
    if(pid)
    {
        //是父进程
        //关闭父进程读fd0
        //
        close(pipe_fd[0]);
        close(sock);
    }
    else
    {
        close(pipe_fd[1]);
    }
}


void Client::Start()
{
    struct epoll_event events[2];
    //连接服务器
    Connect();

    pid = fork();
    if(pid<0)
    {
        perror("fork error");
        close(sock);
        exit(-1);
    }
    else if(pid ==0)
    {
        //子进程
        //写数据，关闭父读数据
        close(pipe_fd[0]);

        cout<<"input 'exit' to exit room" <<endl;
        //如果子进程发送数据到服务器
        while(isClientWork)
        {
            bzero(&message,BUF_SIZE);
            fgets(message,BUF_SIZE,stdin);

            if(strncasecmp(message,EXIT,strlen(EXIT)) ==0 )
            {
                isClientWork = false;
            }
            //没有离开
            else
            {
                //子进程写数据fd1
                //
                if(write(pipe_fd[1],message,strlen(message)-1)<0)
                {
                    perror("fork error");
                    exit(-1);
                }
            }
        }
    }
    //父进程读pipefd0从管道
    //
    else
    {
        //关闭子进程写
        close(pipe_fd[1]);

        while(isClientWork)
        {
            int epoll_events_cont = epoll_wait(epfd,events,2,-1);

            for(int i=0;i<epoll_events_cont;++i)
            {
                bzero(&message,BUF_SIZE);

                //如果消息从服务器来的
                if(events[i].data.fd==sock)
                {
                    int ret = recv(sock,message,BUF_SIZE,0);
                    if(ret ==0)
                    {
                        cout<<"server closed connection："<<sock
                            <<endl;
                        close(sock);
                        isClientWork = false;
                    }
                    else
                    {
                        cout<<message<<endl;
                    }

                }
                //
                else
                {
                    //父进程处理从管道来的数据
                    int ret = read(events[i].data.fd,message,BUF_SIZE);
                    if(ret == 0)
                    {
                        isClientWork = false;
                    }
                    else
                    {
                        send(sock,message,BUF_SIZE,0);
                    }
                }
            }
        }
    }

    Close();
}
