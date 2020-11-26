//
// Created by teamat on 2020-01-25.
//

#include "Socket.h"

using namespace std;

network::Socket::Socket(int p_type, int p_buflen, const std::string& p_port, int p_family, int p_socktype, int p_protocol) :
    m_buflen(p_buflen), m_port(p_port), m_wsaData(), m_connectSocket(INVALID_SOCKET), m_result(NULL), m_recvbuf{0},
    m_ptr(NULL), m_hints()

{
    if (p_type == CLIENT)
    {
        m_iResult = WSAStartup(MAKEWORD(2, 2), &m_wsaData);
        if (m_iResult != 0)
        {
            cout << "WSAStartup failed with error: " << m_iResult << endl;
            throw invalid_argument("WSAStartup failed");
        }
        ZeroMemory(&m_hints, sizeof(m_hints));
        m_hints.ai_family = p_family;
        m_hints.ai_socktype = p_socktype;
        m_hints.ai_protocol = p_protocol;
        // Resolve the server address and port
        m_iResult = getaddrinfo(NULL, p_port.c_str(), &m_hints, &m_result);
        if (m_iResult != 0)
        {
            cout << "getaddrinfo failed with error: " << m_iResult << endl;
            WSACleanup();
            throw invalid_argument("getaddrinfo failed");
        }
        // Attempt to connect to an address until one succeeds
        for (m_ptr = m_result; m_ptr != NULL; m_ptr = m_ptr->ai_next)
        {
            // Create a SOCKET for connecting to server
            m_connectSocket = socket(m_ptr->ai_family, m_ptr->ai_socktype,
                                     m_ptr->ai_protocol);
            if (m_connectSocket == INVALID_SOCKET)
            {
                cout << "socket failed with error: " << WSAGetLastError() << endl;
                WSACleanup();
                throw invalid_argument("socket failed");
            }

            // Connect to server.
            cout << "Connect attempt " << m_ptr->ai_addr << ", " << m_ptr->ai_addrlen << endl;
            m_iResult = connect(m_connectSocket, m_ptr->ai_addr, (int) m_ptr->ai_addrlen);
            if (m_iResult == SOCKET_ERROR)
            {
                closesocket(m_connectSocket);
                m_connectSocket = INVALID_SOCKET;
                continue;
            }
            cout << m_ptr->ai_family << " " << m_ptr->ai_socktype << " " << m_ptr->ai_protocol << endl;
            break;
        }
        freeaddrinfo(m_result);
        if (m_connectSocket == INVALID_SOCKET)
        {
            cout << "Unable to connect to server!" << endl;
            WSACleanup();
            throw invalid_argument("server failed");
        }
    }
}

network::Socket::~Socket()
{
    closesocket(m_connectSocket);
    WSACleanup();
}


void network::Socket::sendGenericSignal(double p_freq)
{
    // generate signal
    unsigned long time_stamp = 0;
    unsigned long count_sent = 0;
    double signal = 0;
    double freq = p_freq;
    chrono::high_resolution_clock::time_point t0 = chrono::high_resolution_clock::now();
    while (1)
    {
        time_stamp ++;
        if (delta_t(t0, chrono::high_resolution_clock::now()) >= 1/freq)
        {
            count_sent ++;
            t0 = chrono::high_resolution_clock::now();
            signal = sin(time_stamp/10.0);
            ostringstream os;
            os << signal;
            const string sendstr = os.str();
            const char* sendchar = sendstr.c_str();
            cout <<count_sent << " " << sendchar << endl;
            m_iResult = send(m_connectSocket, sendchar, (int)strlen(sendchar), 0 );
            if (m_iResult == SOCKET_ERROR) {
                cout << "send failed with error: " << WSAGetLastError() << endl;
                closesocket(m_connectSocket);
                WSACleanup();
                throw invalid_argument("send failed");
            }
        }
    }
}

void network::Socket::synchronousSendRecvExample(double p_freq)
{
    // generate signal
    unsigned long time_stamp = 0;
    unsigned long count_sent = 0;
    double signal = 0;
    chrono::high_resolution_clock::time_point t0 = chrono::high_resolution_clock::now();
    while (1)
    {
        time_stamp ++;
        if (delta_t(t0, chrono::high_resolution_clock::now()) >= 1/p_freq)
        {
            count_sent ++;
            t0 = chrono::high_resolution_clock::now();
            signal = sin(count_sent/p_freq);
            ostringstream os;
            os << signal;
            const string sendstr = os.str();

            const char* sendchar = sendstr.c_str();
//            cout <<count_sent << " " << sendchar << endl;
            m_iResult = send(m_connectSocket, sendchar, (int)strlen(sendchar), 0 );
            if (m_iResult == SOCKET_ERROR) {
                cout << "send failed with error: " << WSAGetLastError() << endl;
                closesocket(m_connectSocket);
                WSACleanup();
                throw invalid_argument("send failed");
            }
            try
            {
                m_iResult = recv(m_connectSocket, m_recvbuf, m_buflen, 0);
                string str_recv = m_recvbuf;
                cout << str_recv << endl;
            }
            catch ( ... )
            {
                cout << "no data recieved" << endl;
            }
        }
    }
}

void network::Socket::synchronousSendRecv(void)
{

}
