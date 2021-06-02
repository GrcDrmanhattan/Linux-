CC = g++
CFLAGS = -std=c++11

all: clientMain.cpp serverMain.cpp server.o client.o
	$(CC) $(CFLAGS) serverMain.cpp server.o -o chatroom_server
	$(CC) $(CFLAGS) clientMain.cpp client.o -o chatroom_client

server.o: server.cpp server.hpp common.hpp
	$(CC) $(CFLAGS) -c server.cpp

client.o: client.cpp client.hpp common.hpp
	$(CC) $(CFLAGS) -c client.cpp

client.i:
	g++ client.cpp -E -o client.i

client.s:
	g++ client.cpp -S -o client.s

clean:
	rm -f *.o chatroom_client chatroom_server


