import java.io.*;
import java.net.*;

public class Client {
    static volatile boolean running = true;

    public static void main(String args[]) throws IOException {
        // connecting to the server
        Socket socket = new Socket("localhost", 9090);
        //in and outs
        PrintWriter out = new PrintWriter(socket.getOutputStream(), true);

        BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));

        BufferedReader keyIn = new BufferedReader(new InputStreamReader(System.in));

        String message;
        String name;
        // prompt for asking the user name
        System.out.print("Enter name: ");
        name = keyIn.readLine();
        out.println(name);

        //if name is taken, user can enter another name, until name is not taken
        String response = in.readLine();
        while (response.equals("Name taken, enter a different name: ")) {
            System.out.print("Name taken, enter a different name: ");
            name = keyIn.readLine();
            out.println(name);
            response = in.readLine();
        }
        // running threads for response
        Thread t = new Thread(new Runnable() {
            public void run() {
                try {
                    while (true) {
                        String response = in.readLine();
                        if (response == null)
                            break;
                        System.out.println(response);
                    }
                } catch (IOException e) {
                }
            }
        });
        t.start();
        // exits the user from the server socket when user types "exit"
        while (true) {
            System.out.print("Enter message: ");
            message = keyIn.readLine();
            out.println(message);
            if (message.equals("exit")) {
                running = false;
                break;
            }
        }
        //close client
        socket.close();
        System.out.println("Client closed.");

    }
}
