#define WIN32_LEAN_AND_MEAN

#ifdef _WIN32
    #include <windows.h>
    #include <winsock2.h>
#else
#include <unistd.h>
#include <arpa/inet.h>
#include <string.h>
#define SOCKET int
#define INVALID_SOCKET (SOCKET)(~0)
#define SOCKET_ERROR           (-1)
#endif

#include <stdio.h>
#include <iostream>
// #include <thread>//MinGW GCC当前仍缺少标准C ++ 11线程类的实现。
#include <mingw.thread.h>


// #pragma comment(lib,"ws2_32.lib")
// using namespace std;

enum CMD
{
    CMD_LOGIN,
    CMD_LOGIN_RESULT,
    CMD_LOGOUT_RESULT,
    CMD_LOGOUT,
    CMD_ERROR,
    CMD_NEW_USER_JOIN
};
struct DataHeader
{
    short dataLength;//数据长度
    short cmd;//命令，描述数据作用
};

//数据包体
struct Login:public DataHeader
{
    Login()
    {
        dataLength = sizeof(Login);
        cmd = CMD_LOGIN;
    }
    char userName[32];
    char Password[32];
};

struct LoginResult: public DataHeader
{
    LoginResult()
    {
        dataLength = sizeof(LoginResult);
        cmd = CMD_LOGIN_RESULT;
        result = 0;//正常
    }
   int result;
};

struct Logout: public DataHeader
{
    Logout()
    {
        dataLength = sizeof(Logout);
        cmd = CMD_LOGOUT;
    }
    char userName[32];
};

struct LogoutResult:public DataHeader
{
    LogoutResult()
    {
        dataLength = sizeof(LogoutResult);
        cmd = CMD_LOGOUT_RESULT;
        result =0;
    }
   int result;
};

struct NewUserJoin:public DataHeader
{
    NewUserJoin()
    {
        dataLength = sizeof(NewUserJoin);
        cmd = CMD_NEW_USER_JOIN;
        sock =0;
    }
   int sock;
};

int processor(SOCKET _cSock)
{
    //缓冲区
    char szRecv[4096]={};//固定长度数据
    //5 接收客户端数据请求
    int nLen=(int)recv(_cSock,(char*)&szRecv,sizeof(DataHeader),0);
    DataHeader* header=(DataHeader*)szRecv;
    if(nLen <= 0)
    {
        printf("yu服务器断开连接，任务结束 \n");
        return -1;
    }
    switch (header->cmd)
    {
        case CMD_LOGIN_RESULT:
        {
            recv(_cSock,szRecv+sizeof(DataHeader), header->dataLength -sizeof(DataHeader),0);
            LoginResult* login =(LoginResult*)szRecv;
            printf("收到服务端消息：CMD_LOGIN_RESULT,数据长度：%d\n",login->dataLength);
        }
        break;
        case CMD_LOGOUT_RESULT:
        {
            recv(_cSock,szRecv+sizeof(DataHeader), header->dataLength -sizeof(DataHeader),0);
            LogoutResult* logout =(LogoutResult*)szRecv;
            printf("收到服务端消息：CMD_LOGOUT_RESULT,数据长度：%d\n",logout->dataLength);
        }
        break;
        case CMD_NEW_USER_JOIN:
        {
            recv(_cSock,szRecv+sizeof(DataHeader), header->dataLength -sizeof(DataHeader),0);
            NewUserJoin* uerJoin =(NewUserJoin*)szRecv;
            printf("收到服务端消息：CMD_NEW_USER_JOIN,数据长度：%d\n",uerJoin->dataLength);
        }
        break;
    }
     return 0;
}

bool g_bRun = true;

void cmdThread(SOCKET _sock)//用来在cmd输入命令
{
    while (true)
    {
        char cmdBuf[256]  = {};
        scanf("%s",cmdBuf);
        if( 0 == strcmp(cmdBuf,"exit"))
        {
            g_bRun = false;
            printf("退出cmdThread\n");
            break;
        }
        else if(0 == strcmp(cmdBuf,"login"))
        {
            Login login;
            strcpy(login.userName,"yxl");
            strcpy(login.Password,"yxldemima");
            send(_sock,(const char*)&login,sizeof(Login),0);
        }
        else if(0 == strcmp(cmdBuf,"logout"))
        {
            Logout logout;
            strcpy(logout.userName,"yxl");
            send(_sock,(const char*)&logout,sizeof(Logout),0);
        }
        else
        {
            printf("不支持的命令\n");
        }
    }
}

int main()
{
#ifdef _WIN32

    WORD ver = MAKEWORD(2,2);
    WSADATA dat;
    WSAStartup(ver,&dat);
#endif

    //
    SOCKET _sock= socket(AF_INET,SOCK_STREAM,0);
    if(INVALID_SOCKET == _sock)
    {
        printf("sock fail\n");
    }
    //2 连接服务器
    sockaddr_in _sin={};
    _sin.sin_family=AF_INET;
    _sin.sin_port = htons(4566);
#ifdef _WIN32
    _sin.sin_addr.S_un.S_addr = inet_addr("192.168.0.105");//连接的服务器ip
#else
    _sin.sin_addr.s_addr = inet_addr("192.168.0.103");
#endif

    int ret = connect(_sock,(sockaddr*)&_sin,sizeof(sockaddr_in));
    if(SOCKET_ERROR == ret)
    {
        printf("connect fail\n");
    }
    else
    {
        printf("连接成功\n");
    }
   //启动线程
    std::thread t1(cmdThread,_sock);
    t1.detach();//和主线程分离

    while (g_bRun)
    {
        fd_set fdReads;
        FD_ZERO(&fdReads);
        FD_SET(_sock,&fdReads);
        timeval t={1,0};
        int ret = select(_sock+1,&fdReads,0,0,&t);
        if(ret <0)
        {
            printf("select 任务结束1\n");
            break;
        }

        if(FD_ISSET(_sock,&fdReads))
        {
            FD_CLR(_sock,&fdReads);
           if(-1== processor(_sock))
           {
                printf("select 任务结束2\n");
                break;
           }
        }
       //线程
        
        // printf("空闲时间处理其他\n");
       
    }
    //了
   //7 关闭
    
#ifdef _WIN32
    //
    closesocket(_sock);
    WSACleanup();
#else
    close(_sock);
#endif

    printf("退出\n");
    getchar();
    return 0;
}
