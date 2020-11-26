//
// Created by teamat on 2020-01-25.
//

#ifndef WINDOWS_CPP_CLIENT_SOCKET_SOCKET_H
#define WINDOWS_CPP_CLIENT_SOCKET_SOCKET_H

#define WIN32_LEAN_AND_MEAN

#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iostream>
#include <sstream>
#include <string>
#include <cmath>
#include <chrono>
#include <ctime>
#include "utils.h"


// Need to link with Ws2_32.lib, Mswsock.lib, and Advapi32.lib
#pragma comment (lib, "Ws2_32.lib")
#pragma comment (lib, "Mswsock.lib")
#pragma comment (lib, "AdvApi32.lib")

namespace network
{
    class Socket
    {
    public:
        Socket(int p_type, int p_buflen=512, const std::string& p_port="5005", int p_family=AF_INET, int p_socktype=SOCK_STREAM, int p_protocol=IPPROTO_TCP);
        ~Socket();
        void sendGenericSignal(double p_freq=100);
        void synchronousSendRecv(void);
        void synchronousSendRecvExample(double p_freq=100);
        enum TYPE{CLIENT=0, SERVER=1};

    private:
    int m_buflen;
    std::string m_port;
    WSADATA m_wsaData;
    SOCKET m_connectSocket;
    addrinfo *m_result;
    addrinfo *m_ptr;
    addrinfo m_hints;
    int m_iResult;
    char m_recvbuf[512];


//    char recvbuf[m_DEFAULT_BUFLEN];
//    int recvbuflen = DEFAULT_BUFLEN;


    };
}



#endif //WINDOWS_CPP_CLIENT_SOCKET_SOCKET_H
