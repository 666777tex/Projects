import java.io.*;
import java.net.*;
// handles each client and creates a new thread 
import java.util.ArrayList;
import java.util.List;

public class ClientHandler implements Runnable {
    Socket clientSocket;
    String name;
    BufferedReader in;
    PrintWriter out;

    ClientHandler(Socket clientSocket) {
        this.clientSocket = clientSocket;
    }

    public void sendMessage(String message) {
        out.println(message);
    }

    public void broadcast(String message) {
        synchronized (Server.clients) {
            for (ClientHandler client : Server.clients) {
                client.sendMessage(message);
            }
        }
    }

    public boolean checkDupe(String name) {
        synchronized (Server.clients) {
            for (ClientHandler client : Server.clients) {
                if (name.equals(client.name)) {
                    return true;
                }
            }
        }
        return false;
    }

    public void handleCommands(String message) throws IOException {
        String[] parts = message.split(" ");
        String command = parts[0];

        switch (command) {
            case ("/ping"):
                sendMessage("pong");
                break;
            case ("/set"):
                CacheEntry setCache = new CacheEntry();
                String setKey = parts[1];
                String setValue = parts[2];
                setCache.value = setValue;
                if (parts.length >= 4) {
                    int seconds = Integer.parseInt(parts[3]);
                    Long milli = Long.valueOf(seconds * 1000);
                    setCache.expiresAt = System.currentTimeMillis() + milli;
                }
                synchronized (Server.store) {
                    Server.store.put(setKey, setCache);
                }
                Server.save();
                sendMessage("Added");
                break;
            case ("/get"):
                String getKey = parts[1];
                synchronized (Server.store) {
                    if (Server.store.containsKey(getKey)) {
                        if (Server.store.get(getKey).expiresAt != null
                                && System.currentTimeMillis() > Server.store.get(getKey).expiresAt) {
                            Server.store.remove(getKey);
                        } else {
                            CacheEntry got = Server.store.get(getKey);
                            sendMessage(got.value);
                        }
                    }
                    if (!Server.store.containsKey(getKey)) {
                        sendMessage("nil");
                    }
                }
                break;
            case ("/delete"):
                synchronized (Server.store) {
                    String deleteKey = parts[1];
                    if (Server.store.containsKey(deleteKey)) {
                        Server.store.remove(deleteKey);
                        sendMessage("Deleted");
                    } else {
                        sendMessage("nil");
                    }
                }
                Server.save();
                break;
            case ("/keys"):
                synchronized (Server.store) {
                    if (Server.store.isEmpty()) {
                        sendMessage("no keys");
                    } else {

                        List<String> expired = new ArrayList<>();

                        for (String key : Server.store.keySet()) {
                            if (Server.store.get(key).expiresAt != null
                                    && Server.store.get(key).expiresAt < System.currentTimeMillis()) {
                                expired.add(key);
                            } else {
                                sendMessage(key);
                            }
                        }
                        for (String eK : expired) {
                            if (Server.store.containsKey(eK)) {
                                Server.store.remove(eK);
                            }
                        }

                    }
                }
                break;
            default:
                sendMessage("Unknown command: " + command);
        }
    }

    public void run() {
        try {
            in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));
            out = new PrintWriter(clientSocket.getOutputStream(), true);
            name = in.readLine();
            boolean added = false;
            while (!added) {
                synchronized (Server.clients) {
                    if (!checkDupe(name)) {
                        Server.clients.add(this);
                        added = true;
                    }
                }
                if (!added) {
                    out.println("Name taken, enter a different name: ");
                    name = in.readLine();
                }
            }

            broadcast(name + " connected.");

            while (true) {
                String message = in.readLine();
                if (message == null)
                    break;
                if (message.equals("exit")) {
                    break;
                }
                if (message.startsWith("/")) {
                    handleCommands(message);
                } else {
                    System.out.println(name + " says: " + message);
                    String fullMessage = name + ": " + message;
                    broadcast(fullMessage);
                }

            }

        } catch (IOException e) {
            System.out.println("Connection error: " + e.getMessage());
        } finally {
            synchronized (Server.clients) {
                Server.clients.remove(this);
            }
            if (name != null) {
                broadcast(name + " disconnected");
                System.out.println(name + " disconnected");
            }
            try {
                clientSocket.close();

            } catch (IOException e) {
            }
        }
    }

}