from dataclasses import dataclass
import os
import sys
import termios
import time
import tty

from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus as MotorBus

PORT = "/dev/tty.usbmodem<YOUR PORT>" # Put YOUR port here
MOTOR_IDS = [1, 2, 3, 4, 5, 6]
# MOTOR_IDS = [1, 2, 3, 5, 4, 6] # Mine were installed w/ 4 & 5 swapped... :-/

MOTOR_MODEL = "sts3215"
MOTOR_MODELS = [MOTOR_MODEL] * len(MOTOR_IDS)
N_MOTORS = len(MOTOR_IDS)
BAUDRATE = 1_000_000
SCAN_IDS = list(range(1, 10)) # Motor IDs to check for; SO100 only has 6, but higher possible if misconfigured
PID_P, PID_I, PID_D = 16, 0, 32
ACCELERATION = 20 # HF is using 256

SEQUENCES = {
    "high_five": [ # [[position, wait_secs], ...]
        [[3006, 1285, 2284, 1458, 1029, 2370], 2.0],
        [[3006, 2227, 1397, 1458, 1029, 2370], 2.0],
    ],
}


@dataclass
class Joint:
    range_low: int # lowest servo position
    range_high: int # highest servo position
    resolution: int # n steps
    home: int # position when homed
    limit_low: int # lowest joint position; >= range_low
    limit_high: int # highest joint position; <= range_high


JOINTS = [
    #     l  h     res   home  lim_l lim_h
    Joint(0, 4096, 4096, 2100,  990, 2980), # shoulder pan (1)
    Joint(0, 4096, 4096, 1070, 1070, 3030), # shoulder lift (2)
    Joint(0, 4096, 4096, 2975, 1090, 3020), # elbow flex (3)

    Joint(0, 4096, 4096, 2500, 1060, 2790), # wrist flex (4)
    Joint(0, 4096, 4096, 1022,  590, 4000), # wrist roll (5)
    Joint(0, 4096, 4096, 2374, 1950, 3260), # gripper (6)
]

HOME_POSITION = [x.home for x in JOINTS]
SHUTDOWN_SEQUENCE = [
    [2095, 1482, 2347, 1460, 1022, 2368], # safe upright pose
    HOME_POSITION,
]


def clamp(low, val, high):
    if val > high:
        return high
    if val < low:
        return low
    return val


def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def announce_active():
    os.system('say -v "Zarvox" -r 200 "Robot... activated"')


def announce_shutdown():
    os.system('say -v "Zarvox" -r 200 "Robot... Shutting Down"')


def announce_fetch():
    os.system('say -v "Zarvox" -r 200 "EV... Time to play fetch"')


def write_values(motor_bus, data_name, values, motor_ids):
    '''Write value_i to motor_i'''
    assert len(values) == len(motor_ids)
    motor_bus.write_with_motor_ids(
        motor_models=[MOTOR_MODEL] * len(values), # All same for this arm
        motor_ids=motor_ids,
        data_name=data_name,
        values=values,
    )


def write_value(motor_bus, data_name, value):
    '''Write same value to all motors'''
    write_values(
        motor_bus,
        data_name=data_name,
        values=[value] * len(MOTOR_IDS),
        motor_ids=MOTOR_IDS,
    )


def read_value(motor_bus, data_name):
    '''Read value from ALL motors'''
    return motor_bus.read_with_motor_ids(
        motor_models=MOTOR_MODELS,
        motor_ids=MOTOR_IDS,
        data_name=data_name,
    )


def make_safe(position):
    '''Ensure joint positions are within limits'''
    assert len(position) == len(MOTOR_IDS)
    safe_pos = position[:]
    for motor_i in range(len(position)):
        joint_i = JOINTS[motor_i]
        safe_pos[motor_i] = clamp(joint_i.limit_low, position[motor_i], joint_i.limit_high)
    return safe_pos


def get_position(motor_bus) -> list[int]:
    '''Get position of all joints'''
    return read_value(motor_bus, "Present_Position")


def set_goal(motor_bus, position):
    '''Set goal state for motors

    Note that this can target unsafe position, use make_safe if safety required
    '''
    assert len(position) == len(MOTOR_IDS)
    write_values(motor_bus, data_name="Goal_Position", values=position, motor_ids=MOTOR_IDS)


def set_torque(motor_bus, enable=False):
    '''1 is on; 0 is off'''
    write_value(motor_bus, data_name="Torque_Enable", value=1 if enable else 0)


def prepare(motor_bus):
    '''Prepare bot for automated movement'''
    # Disable torque -> configure motors -> enable torque
    set_torque(motor_bus, enable=False)
    for field, value in (
        ("Mode", 0), # Hold position mode (it's the default anyways)
        ("P_Coefficient", PID_P),
        ("I_Coefficient", PID_I),
        ("D_Coefficient", PID_D),
        ("Maximum_Acceleration", ACCELERATION), # XXX: Write lock needed?
        ("Acceleration", ACCELERATION),
    ):
        write_value(motor_bus, data_name=field, value=value)
    set_torque(motor_bus, enable=True)

    # Move to home position
    set_goal(motor_bus, HOME_POSITION)
    announce_active()
    print("Arm is ready for action!")


def shut_down(motor_bus):
    print("Shutting down...")
    announce_shutdown()

    # Safely move to home position
    for pos in SHUTDOWN_SEQUENCE:
        set_goal(motor_bus, make_safe(pos))
        time.sleep(3) # Wait until moved to goal XXX: Make fxn to await close enough

    # Turn off torque
    set_torque(motor_bus, enable=False)

    # Tell motor bus we're done
    motor_bus.disconnect()
    print("Motor bus disconnected")


def run_routine(motor_bus, name):
    prepare(motor_bus)
    info(motor_bus)
    if name not in SEQUENCES:
        raise KeyError(f"No sequence with name {name}")
    for position, wait_secs in SEQUENCES[name]:
        print("keyframe:", position)
        set_goal(motor_bus, make_safe(position))
        time.sleep(wait_secs)


def manual_control(motor_bus):
    prepare(motor_bus)

    # Instructions
    print("\nManual Control")
    print("Q - Quit (uppercase!)")
    print("w/a/s/d/q/e - Move arm")
    print("j/l/k/i/u/o - Move hand")
    print()

    incr = 20 # amount moved each input loop
    torque_active = True # Starts w/ torque since prepare
    char_config = {
        # Shoulder/Arm
        'a': (1, -1), 'd': (1, +1),
        's': (3, +1), 'w': (3, -1),
        'q': (2, -1), 'e': (2, +1),
        'f': (2, -1), 'r': (2, +1), # same as q/e, for convenience

        # Wrist/Hand
        'j': (5, +1), 'l': (5, -1),
        'k': (4, +1), 'i': (4, -1),
        'u': (6, -1), 'o': (6, +1),
    }
    position = HOME_POSITION[:] # init to home
    set_goal(motor_bus, make_safe(position))
    while True:
        char = getch()
        if char in char_config: # move w/ keyboard
            motor_id, direc = char_config[char]
            motor_idx = motor_id - 1
            position[motor_idx] = position[motor_idx] + direc*incr
            set_goal(motor_bus, make_safe(position))
        elif char == "t": # toggle torque
            torque_active = not torque_active
            curr_position = get_position(motor_bus)
            position = curr_position
            set_torque(motor_bus, enable=torque_active)
        elif char == 'Q':
            break
        else:
            continue
        print(position, f"torque:{torque_active}")


def info(motor_bus):
    '''Display info about bus/joints'''
    baud_rate = motor_bus.port_handler.getBaudRate()
    motors_identified = motor_bus.find_motor_indices(possible_ids=SCAN_IDS)
    fields = [
        "Mode",
        "P_Coefficient",
        "I_Coefficient",
        "D_Coefficient",
        "Maximum_Acceleration",
        "Acceleration",
        "Torque_Enable",
        "Offset",
        "Present_Position",
        "Goal_Position",
    ]
    values = []
    for field in fields:
        values.append(read_value(motor_bus, field))

    print("\n[Bus Info]")
    fields = ["Bus Baud", "Motor IDs Detected"] + fields
    values = [baud_rate, motors_identified] + values
    for field, val in zip(fields, values):
        print(f"{field}:".ljust(30), val)
    print()


def monitor_position(motor_bus):
    print("\nPosition Monitoring")
    prev_positions = [0] * N_MOTORS
    while True:
        time.sleep(1)
        positions = get_position(motor_bus)
        positions_delta = [positions[i] - prev_positions[i] for i in range(N_MOTORS)]
        prev_positions = positions
        print(
            "[P]",
            " ".join([str(x).rjust(6) for x in positions]),
            "[Δ]".rjust(6),
            " ".join([str(x).rjust(6) for x in positions_delta]),
        )


def menu(choices):
    '''
    choices ~ (("Menu Option 1", lambda: do_something(...)), ...)
    '''
    # Get choice from user
    while True:
        print("\nWhat do you want to do?")
        for i, info in enumerate(choices):
            print(f"{i+1}. {info[0]}")
        choice = input(f"\nEnter your choice (1-{len(choices)}): ")
        if choice in [str(x) for x in range(1, len(choices) + 1)]:
            break

    # Run callback corresponding to choice
    choices[int(choice)-1][1]()


def main_menu(motor_bus):
    # Generate routine submenu choices
    routine_choices = []
    for seq_name in SEQUENCES:
        cback = lambda x=seq_name: run_routine(motor_bus, x) # noqa; x=seqname required to capture!
        routine_choices.append((seq_name, cback))

    # Main menu
    main_choices = (
        ("Manual control", lambda: manual_control(motor_bus)),
        ("Run routine", lambda: menu(routine_choices)),
        ("Monitor position", lambda: monitor_position(motor_bus)),
        ("Quit", lambda: print("Goodbye!")),
    )
    menu(main_choices)


if __name__ == "__main__":

    # Initialize the MotorBus
    # - "motors" info is arbitrary here, we're using raw read/write methods
    #   - XXX: These "raw" MotorBus methods should just be class methods...
    motor_bus = MotorBus(port=PORT, motors={"motor": (-1, MOTOR_MODEL)})
    motor_bus.connect()

    try:
        # Display info first
        info(motor_bus)

        # Choose/start task
        main_menu(motor_bus)

    finally:
        shut_down(motor_bus)