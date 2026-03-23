#include "WiFiS3.h"

// Wi-Fi credentials 
char ssid[] = "NETWORK_NAME";
char pass[] = "NETWORK_PASSWORD";
int status = WL_IDLE_STATUS;

// Set up server on port 80
WiFiServer server(80);

// Hardware pins
const int ledPin = 13;      // Built-in LED on Uno R4
const int relay1Pin = 8;    // Example: Lights
const int relay2Pin = 9;    // Example: Ventilator

// State tracking
bool lightsOn = false;
bool fanOn = false;

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    ; // wait for serial port to connect
  }

  // Initialize pins
  pinMode(ledPin, OUTPUT);
  pinMode(relay1Pin, OUTPUT);
  pinMode(relay2Pin, OUTPUT);
  
  // Set initial states
  digitalWrite(ledPin, LOW);
  digitalWrite(relay1Pin, LOW);
  digitalWrite(relay2Pin, LOW);

  // Attempt to connect to WiFi network
  if (WiFi.status() == WL_NO_MODULE) {
    Serial.println("Communication with WiFi module failed!");
    while (true);
  }

  String fv = WiFi.firmwareVersion();
  if (fv < WIFI_FIRMWARE_LATEST_VERSION) {
    Serial.println("Please upgrade the firmware");
  }

  Serial.print("Attempting to connect to SSID: ");
  Serial.println(ssid);
  
  status = WiFi.begin(ssid, pass);
  while (status != WL_CONNECTED) {
    delay(1000);
    Serial.print(".");
    status = WiFi.begin(ssid, pass);
  }

  Serial.println("\nConnected to WiFi!");
  printWifiStatus();

  // Start the web server
  server.begin();
}

void loop() {
  // Listen for incoming clients
  WiFiClient client = server.available();
  if (client) {
    Serial.println("New client connection...");
    String currentLine = "";
    
    // Read the HTTP request
    while (client.connected()) {
      if (client.available()) {
        char c = client.read();
        Serial.write(c); // Print to serial monitor
        
        if (c == '\n') {
          // If a newline is received, the line is complete.
          // An empty line means the headers are done.
          if (currentLine.length() == 0) {
            // Send standard HTTP response header
            client.println("HTTP/1.1 200 OK");
            client.println("Content-type:text/plain");
            client.println("Connection: close");
            client.println();
            client.println("Command received.");
            break;
          } else {
            // Parse the GET request URL
            if (currentLine.startsWith("GET /LIGHTS_TOGGLE")) {
              lightsOn = !lightsOn;
              digitalWrite(relay1Pin, lightsOn ? HIGH : LOW);
              digitalWrite(ledPin, lightsOn ? HIGH : LOW);
              Serial.println("Action: Toggled Lights");
            } 
            else if (currentLine.startsWith("GET /FAN_TOGGLE")) {
              fanOn = !fanOn;
              digitalWrite(relay2Pin, fanOn ? HIGH : LOW);
              Serial.println("Action: Toggled Fan");
            }
            else if (currentLine.startsWith("GET /ALL_ON")) {
              lightsOn = true;
              fanOn = true;
              digitalWrite(relay1Pin, HIGH);
              digitalWrite(relay2Pin, HIGH);
              digitalWrite(ledPin, HIGH);
              Serial.println("Action: All ON");
            }
            else if (currentLine.startsWith("GET /ALL_OFF")) {
              lightsOn = false;
              fanOn = false;
              digitalWrite(relay1Pin, LOW);
              digitalWrite(relay2Pin, LOW);
              digitalWrite(ledPin, LOW);
              Serial.println("Action: All OFF");
            }
            else if (currentLine.startsWith("GET /DEVICE_3")) {
              Serial.println("Action: Pinged Device 3 (Pinky Open)");
            }
            
            // Clear currentLine to read the next line
            currentLine = "";
          }
        } 
        else if (c != '\r') {
          // add it to the end of the currentLine
          currentLine += c;
        }
      }
    }
    // Give the web browser / client time to receive the data
    delay(10);
    // Close the connection
    client.stop();
  }
}

void printWifiStatus() {
  Serial.print("SSID: ");
  Serial.println(WiFi.SSID());

  IPAddress ip = WiFi.localIP();
  Serial.print("IP Address: ");
  Serial.println(ip);
}
