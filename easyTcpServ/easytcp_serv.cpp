#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
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
#include <vector>
#include <thread>//MinGW GCC当前仍缺少标准C ++ 11线程类的实现。
//#include <mingw.thread.h>
//using namespace std; 导致bind函数报错在xcode
//测试vscode svn
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

std::vector<SOCKET> g_clients;//全局变量

int processor(SOCKET _cSock)
{
    //缓冲区
    char szRecv[4096]={};//固定长度数据
    //5 接收客户端数据请求
    int nLen=(int)recv(_cSock,(char*)&szRecv,sizeof(DataHeader),0);
    DataHeader* header=(DataHeader*)szRecv;
    if(nLen <= 0)
    {
        printf("客户端<socket=%d>已退出任务结束 \n",_cSock);
        return -1;
    }

    switch (header->cmd)
    {
        case CMD_LOGIN:
        {
            
            recv(_cSock,szRecv+sizeof(DataHeader), header->dataLength -sizeof(DataHeader),0);
            Login* login =(Login*)szRecv;
            printf("客户端<socket=%d>请求：cmd_login,数据长度：%d ,username=%s ,pssword=%s\n",_cSock,login->dataLength,login->userName,login->Password);
                
            //忽略判断用户名密码
            LoginResult ret;
            send(_cSock,(char*)&ret,sizeof(LoginResult),0);
        }
        break;
        case CMD_LOGOUT:
        {
            
            recv(_cSock,szRecv+sizeof(DataHeader),header->dataLength - sizeof(DataHeader),0);
            Logout* logout = (Logout*)szRecv ;
            printf("客户端<socket=%d>请求：cmd_logout,数据长度：%d ,username=%s \n",_cSock,logout->dataLength,logout->userName);

            //忽略判断用户名密码
            LogoutResult ret;
            // send(_cSock,(char*)&header,sizeof(header),0);
            send(_cSock,(char*)&ret,sizeof(ret),0);
        }
        break;
        default:
        {
            DataHeader header={0,CMD_ERROR};
            send(_cSock,(char*)&header,sizeof(header),0);
        }
        break;
    }
     return 0;
}

int main()
{
#ifdef _WIN32
    WORD ver = MAKEWORD(2,2);
    WSADATA dat;
    WSAStartup(ver,&dat);//启动winsocket环境
#endif
    //
    SOCKET _sock=socket(AF_INET,SOCK_STREAM,0);
    sockaddr_in _sin={};
    _sin.sin_family = AF_INET;
    _sin.sin_port = htons(4566);
#ifdef _WIN32
    _sin.sin_addr.S_un.S_addr = INADDR_ANY;//接受任意ip
#else
    _sin.sin_addr.s_addr = INADDR_ANY;
#endif
    if(SOCKET_ERROR == bind(_sock,(sockaddr*)&_sin,sizeof(_sin)))
    {
        printf("bind fail\n");
    }
    else
    {
        printf("绑定成功、\n");
    }
    if(SOCKET_ERROR == listen(_sock,5))
    {
        printf("shibai listen\n");
    }
    else
    {
        printf("listen成功、\n");
    }

    while(true)
    {
        //伯克利套接字socket
        fd_set fdRead;//描述符socket集合
        fd_set fdWrite;
        fd_set fdExp;
        //清理集合
        FD_ZERO(&fdRead);
        FD_ZERO(&fdWrite);
        FD_ZERO(&fdExp);

        FD_SET(_sock,&fdRead);//将套接字加入集合
        FD_SET(_sock,&fdWrite);
        FD_SET(_sock,&fdExp);
        SOCKET maxSock=_sock;
        //size_t 类型不能“--”运算
        for(int n=(int) g_clients.size()-1;n>=0;--n)
        {
            FD_SET(g_clients[n],&fdRead);
            if(maxSock < g_clients[n])
            {
                maxSock = g_clients[n];
            }
        }


        //nfds是一个整数，是指fdset集合中所有描述符（socket）的范围，而不是数量
        //既是所有文件描述符包括客户端和服务器，的最大值+1，在win这个参数无所谓可以写0
        // timeval t = {0,0};
        timeval t={1,0};//最大为等待1秒,时间到没有数据就返回处理其他业务
        //select 查询后如果有操作数据就保留在相应的集合中
        //假如没有时间参数，select会等待有数据才会返回处理，会被阻塞
        int ret= select(maxSock + 1, &fdRead, &fdWrite,&fdExp,&t);
        if(ret<0)
        {
            printf("select 结束\n");
            break;
        }
        if(FD_ISSET(_sock,&fdRead))//判断描述符是否在集合中
        {
            FD_CLR(_sock,&fdRead);

            sockaddr_in clientAddr={};
            int clientaddr_len=sizeof(sockaddr_in);
            SOCKET _cSock=INVALID_SOCKET;
#ifdef _WIN32
            _cSock= accept(_sock,(sockaddr*)&clientAddr,&clientaddr_len);
#else
            _cSock= accept(_sock,(sockaddr*)&clientAddr,(socklen_t*)&clientaddr_len);
#endif
            if(INVALID_SOCKET == _cSock)
            {
                printf(" 无效client socket\n");
            }
            else
            {
                for(int n=(int) g_clients.size()-1;n>=0;--n)
                {
                    NewUserJoin userJoin;
                    send(g_clients[n],(const char*)&userJoin,sizeof(NewUserJoin),0);
                }
                g_clients.push_back(_cSock);
                printf("new client: socket =%d ,ip = %s \n",_cSock,inet_ntoa( clientAddr.sin_addr));
            }
        }

        for(int n=(int) g_clients.size()-1;n>=0;--n)
        {
            if(FD_ISSET(g_clients[n], &fdRead))
            {
                if(-1 == processor(g_clients[n]))
                {
                    auto iter = g_clients.begin()+n;//std::vector<socket>::iterator
                    if(iter != g_clients.end())
                    {
                        g_clients.erase(iter);
                    }
                }
            }
        }
        
        // printf("t设置为1秒后，空闲时间处理其他业务\n");
    }
#ifdef _WIN32
    for(int n=(int) g_clients.size()-1;n>=0;--n)
    {
        closesocket(g_clients[n]);
    }

   //8 关闭
    closesocket(_sock);
    WSACleanup();
#else
    for(int n= (int)g_clients.size()-1;n>=0;--n)
      {
          close(g_clients[n]);
      }

     //8 关闭
      close(_sock);
#endif
    printf("已结束\n");
    getchar();
    return 0;
}
