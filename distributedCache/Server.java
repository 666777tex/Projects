import java.io.*;
import java.net.*;
import java.util.*;
// import com.opencsv.CSVWriter;

import javax.management.RuntimeErrorException;

public class Server {
    static List<ClientHandler> clients = Collections.synchronizedList(new ArrayList<>());
    static Map<String, CacheEntry> store = new HashMap<>();

    public static void main(String args[]) throws IOException {
        // create a server socket on port number 9090
        ServerSocket serverSocket = new ServerSocket(9090);
        File file = new File("store.csv");
        // if file exit, then look for expired and get rid of them, else, create a new
        // csv file
        if (file.exists() && !file.isDirectory()) {
            FileWriter fwriter = new FileWriter(file);
            // read each one (BufferedReader) in the existed csv file and get rid of the
            // expired ones
            try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
                reader.readLine();
                String line;
                while ((line = reader.readLine()) != null) {
                    String[] data = line.split(",");
                    String key = data[0];
                    String value = data[1];
                    String eA = data[2];
                    Long expireAt;
                    if (eA.equals("null")){
                        expireAt = null;
                    } else {
                        expireAt = Long.valueOf(data[2]);

                    }
                    if (expireAt != null && expireAt > System.currentTimeMillis()) {
                        CacheEntry cache = new CacheEntry();
                        cache.value = value;
                        cache.expiresAt = expireAt;
                        store.put(key, cache);
                    }
                }
                fwriter.close();
                for (String key : store.keySet()){

                }
            } catch (IOException e) {
                throw new RuntimeException("Error reading the file ", e);
            }
            fwriter.close();
        } else {
            file.createNewFile();
            FileWriter fwriter = new FileWriter(file);
            fwriter.append("key, value, expireAt");
            fwriter.close();
        }

        System.out.println("Server is running and waiting for client connection...");
        while (true) {
            // Accept incoming client connection
            Socket clientSocket = serverSocket.accept();
            ClientHandler handler = new ClientHandler(clientSocket);
            Thread th = new Thread(handler);
            th.start();

        }

    }
}