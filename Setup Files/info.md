## `.at` file: 
- JSON object that maps a sting command to a corresponding id. The id must be a unique integer greater than 1
- it must loaded by the GUI (`signgui.py`) and the assistive device control program (`pc_control.py`)
- currently hardcoded
- example : `{"copy": 2, "paste": 3, "lclick": 4, "rclick": 5, "up": 6, "down": 7, "right": 8, "left": 9, "tab": 10, "enter": 11}`

## `.asq` file:
- JSON object
- The keys are the user defined sequences
- the values are a boolean (0 or 1) indicating if the sequence is active on loading
- it must be loaded by the GUI
- example: `{"tap": 0, "tap-tap": 1, "swipe-swipe": 1}`

## `.acm` file:
- JSON length-2 array
- first entry is the name of the assiciated `.at` file (without the extension)
- second entry is a JSON object mapping a user defined sequence (from the keys of the `.asq` file) to a string command of the `.at` file
- example: `["at_pc_dev", {"tap": "up", "tap-tap": "down", "swipe-swipe": "right"}]`

## `.sd` file:
- contain the training data 
- format: 
``` python
{
    'first_gesture':
        [
            0,
            [
                pandas.DataFrame(sensor1),
                pandas.DataFrame(sensor2),
                pandas.DataFrame(sensor3),
                pandas.DataFrame(sensor4),
                pandas.DataFrame(sensor5),
                pandas.DataFrame(sensor6),
            ]
        ],
    'second_gesture':
        [
            1,
            [ 
                ... 
            ]
        ],
    ...
}
```
example: 
``` python
{
    'tap':
        [
            0,
            [
                pandas.DataFrame(accel_x),
                pandas.DataFrame(accel_y),
                pandas.DataFrame(accel_z),
                pandas.DataFrame(gyro_x),
                pandas.DataFrame(gyro_x),
                pandas.DataFrame(gyro_x),
            ]
        ],
    'swipe':
        [
            1,
            [ 
                ... 
            ]
        ],
    ...
}
```
- must be loaded by `utils.load_imu_data(file)`