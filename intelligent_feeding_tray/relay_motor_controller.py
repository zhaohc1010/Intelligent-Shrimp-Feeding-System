# Filename: relay_motor_controller.py
# Description: Hardware module dedicated to controlling relays and motors

import serial
import time
import threading
import logging # Use logging instead of print to match app.py

# Get a logger instance
logger = logging.getLogger(__name__)

# ==============================================================================
#  Low-level Relay Driver
# ==============================================================================
class LCUSRelayController:
    """A Python class for controlling USB Relay (LCUS-x,x) series modules."""

    def __init__(self, com_port='COM3', baudrate=9600, relay_channels=4):
        self.ser = None
        try:
            self.ser = serial.Serial(
                port=com_port, baudrate=baudrate, bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=0.5
            )
        except serial.SerialException as e:
            # Use logger.error to record errors
            logger.error(f"❌ Relay Driver Error: Cannot open serial port {com_port}. Error: {e}")

    def _send_command(self, cmd):
        if not self.ser or not self.ser.is_open: return
        self.ser.write(cmd)
        time.sleep(0.05)

    def _generate_command(self, relay, action):
        start_byte, relay_address, action_byte = 0xA0, relay, action
        checksum = (start_byte + relay_address + action_byte) & 0xFF
        return bytes([start_byte, relay_address, action_byte, checksum])

    def turn_on(self, relay=1):
        cmd = self._generate_command(relay, 0x01)
        if cmd: self._send_command(cmd)

    def turn_off(self, relay=1):
        cmd = self._generate_command(relay, 0x00)
        if cmd: self._send_command(cmd)

    def close(self):
        if self.ser and self.ser.is_open: self.ser.close()


# ==============================================================================
#  Motor Controller
# ==============================================================================
class RelayMotorController:
    """
    Controller for motors using LCUS relays.
    - Relay 1: Down
    - Relay 2: Up
    - Relay 3 & 4: Main Motor Power Switch
    """

    def __init__(self, port="COM3"):
        logger.info("🔧 Initializing Relay Motor Controller...")
        self.relay_ctrl = LCUSRelayController(com_port=port)
        self.is_motor_open = False

        if self.relay_ctrl.ser:
            logger.info("✅ Relay Motor Controller connected successfully.")
            # Ensure all relays are off upon initialization
            self.relay_ctrl.turn_off(1)
            self.relay_ctrl.turn_off(2)
            self.relay_ctrl.turn_off(3)
            self.relay_ctrl.turn_off(4)
        else:
            logger.warning("❌ Relay Motor Controller initialization failed. Running in simulation mode.")

    def open_motor(self):
        """Turn ON motor power (Relays 3 and 4)"""
        logger.info("⚡ Turning ON motor power (Relay 3 & 4)...")
        self.relay_ctrl.turn_on(3)
        self.relay_ctrl.turn_on(4)
        self.is_motor_open = True
        logger.info("✅ Motor power is ON.")

    def close_motor(self):
        """Turn OFF motor power (Relays 3 and 4)"""
        logger.info("🔌 Turning OFF motor power (Relay 3 & 4)...")
        self.relay_ctrl.turn_off(3)
        self.relay_ctrl.turn_off(4)
        self.is_motor_open = False
        logger.info("✅ Motor power is OFF.")

    def _timed_move_worker(self, relay_number, duration):
        """Timed task executed in a background thread"""
        logger.info(f"  ⬆️⬇️ Motor action started (Relay {relay_number} ON) for {duration} seconds...")
        self.relay_ctrl.turn_on(relay_number)
        time.sleep(duration)
        self.relay_ctrl.turn_off(relay_number)
        logger.info(f"  🛑 Motor action finished (Relay {relay_number} auto-closed).")

    def move_up(self, duration):
        """Move motor UP for specified time (Relay 2)"""
        if not self.is_motor_open:
            self.open_motor()
            time.sleep(0.5)
        logger.info(f"⚙️ Preparing to move UP for {duration} seconds...")
        thread = threading.Thread(target=self._timed_move_worker, args=(2, float(duration)))
        thread.daemon = True
        thread.start()

    def move_down(self, duration):
        """Move motor DOWN for specified time (Relay 1)"""
        if not self.is_motor_open:
            self.open_motor()
            time.sleep(0.5)
        logger.info(f"⚙️ Preparing to move DOWN for {duration} seconds...")
        thread = threading.Thread(target=self._timed_move_worker, args=(1, float(duration)))
        thread.daemon = True
        thread.start()

    def cleanup(self):
        """Clean up resources on program exit"""
        logger.info("🧹 Closing relay connection...")
        self.close_motor()
        self.relay_ctrl.close()