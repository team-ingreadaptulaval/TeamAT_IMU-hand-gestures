#define WIN32_LEAN_AND_MEAN

#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iostream>
#include <sstream>
#include <cmath>
#include <ctime>
#include <chrono>
#include "utils.h"
#include "Socket.h"


// Need to link with Ws2_32.lib, Mswsock.lib, and Advapi32.lib
#pragma comment (lib, "Ws2_32.lib")
#pragma comment (lib, "Mswsock.lib")
#pragma comment (lib, "AdvApi32.lib")


#define DEFAULT_BUFLEN 512
#define DEFAULT_PORT "5005"

using namespace std;
using namespace network;

int main(int argc, char **argv)  //__cdecl
{

    Socket s(Socket::CLIENT);
    s.sendGenericSignal(10);
    s.synchronousSendRecv();
    return 0;

}