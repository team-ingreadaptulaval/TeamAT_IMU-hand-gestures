/****************************************************************************
** Meta object code from reading C++ file 'myxda.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.9.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../Monitor/awindamonitor_cpp/myxda.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'myxda.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.9.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_MyXda_t {
    QByteArrayData data[11];
    char stringdata0[130];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_MyXda_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_MyXda_t qt_meta_stringdata_MyXda = {
    {
QT_MOC_LITERAL(0, 0, 5), // "MyXda"
QT_MOC_LITERAL(1, 6, 22), // "wirelessMasterDetected"
QT_MOC_LITERAL(2, 29, 0), // ""
QT_MOC_LITERAL(3, 30, 10), // "XsPortInfo"
QT_MOC_LITERAL(4, 41, 17), // "dockedMtwDetected"
QT_MOC_LITERAL(5, 59, 11), // "mtwUndocked"
QT_MOC_LITERAL(6, 71, 18), // "openPortSuccessful"
QT_MOC_LITERAL(7, 90, 14), // "openPortFailed"
QT_MOC_LITERAL(8, 105, 5), // "reset"
QT_MOC_LITERAL(9, 111, 8), // "openPort"
QT_MOC_LITERAL(10, 120, 9) // "scanPorts"

    },
    "MyXda\0wirelessMasterDetected\0\0XsPortInfo\0"
    "dockedMtwDetected\0mtwUndocked\0"
    "openPortSuccessful\0openPortFailed\0"
    "reset\0openPort\0scanPorts"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_MyXda[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       8,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       5,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   54,    2, 0x06 /* Public */,
       4,    1,   57,    2, 0x06 /* Public */,
       5,    1,   60,    2, 0x06 /* Public */,
       6,    1,   63,    2, 0x06 /* Public */,
       7,    1,   66,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       8,    0,   69,    2, 0x0a /* Public */,
       9,    1,   70,    2, 0x0a /* Public */,
      10,    0,   73,    2, 0x09 /* Protected */,

 // signals: parameters
    QMetaType::Void, 0x80000000 | 3,    2,
    QMetaType::Void, 0x80000000 | 3,    2,
    QMetaType::Void, 0x80000000 | 3,    2,
    QMetaType::Void, 0x80000000 | 3,    2,
    QMetaType::Void, 0x80000000 | 3,    2,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 3,    2,
    QMetaType::Void,

       0        // eod
};

void MyXda::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        MyXda *_t = static_cast<MyXda *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->wirelessMasterDetected((*reinterpret_cast< XsPortInfo(*)>(_a[1]))); break;
        case 1: _t->dockedMtwDetected((*reinterpret_cast< XsPortInfo(*)>(_a[1]))); break;
        case 2: _t->mtwUndocked((*reinterpret_cast< XsPortInfo(*)>(_a[1]))); break;
        case 3: _t->openPortSuccessful((*reinterpret_cast< XsPortInfo(*)>(_a[1]))); break;
        case 4: _t->openPortFailed((*reinterpret_cast< XsPortInfo(*)>(_a[1]))); break;
        case 5: _t->reset(); break;
        case 6: _t->openPort((*reinterpret_cast< XsPortInfo(*)>(_a[1]))); break;
        case 7: _t->scanPorts(); break;
        default: ;
        }
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        switch (_id) {
        default: *reinterpret_cast<int*>(_a[0]) = -1; break;
        case 0:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< XsPortInfo >(); break;
            }
            break;
        case 1:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< XsPortInfo >(); break;
            }
            break;
        case 2:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< XsPortInfo >(); break;
            }
            break;
        case 3:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< XsPortInfo >(); break;
            }
            break;
        case 4:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< XsPortInfo >(); break;
            }
            break;
        case 6:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< XsPortInfo >(); break;
            }
            break;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        void **func = reinterpret_cast<void **>(_a[1]);
        {
            typedef void (MyXda::*_t)(XsPortInfo );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&MyXda::wirelessMasterDetected)) {
                *result = 0;
                return;
            }
        }
        {
            typedef void (MyXda::*_t)(XsPortInfo );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&MyXda::dockedMtwDetected)) {
                *result = 1;
                return;
            }
        }
        {
            typedef void (MyXda::*_t)(XsPortInfo );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&MyXda::mtwUndocked)) {
                *result = 2;
                return;
            }
        }
        {
            typedef void (MyXda::*_t)(XsPortInfo );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&MyXda::openPortSuccessful)) {
                *result = 3;
                return;
            }
        }
        {
            typedef void (MyXda::*_t)(XsPortInfo );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&MyXda::openPortFailed)) {
                *result = 4;
                return;
            }
        }
    }
}

const QMetaObject MyXda::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_MyXda.data,
      qt_meta_data_MyXda,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *MyXda::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *MyXda::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_MyXda.stringdata0))
        return static_cast<void*>(const_cast< MyXda*>(this));
    return QObject::qt_metacast(_clname);
}

int MyXda::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 8)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 8;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 8)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 8;
    }
    return _id;
}

// SIGNAL 0
void MyXda::wirelessMasterDetected(XsPortInfo _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void MyXda::dockedMtwDetected(XsPortInfo _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void MyXda::mtwUndocked(XsPortInfo _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}

// SIGNAL 3
void MyXda::openPortSuccessful(XsPortInfo _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 3, _a);
}

// SIGNAL 4
void MyXda::openPortFailed(XsPortInfo _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 4, _a);
}
struct qt_meta_stringdata_MyWirelessMasterCallback_t {
    QByteArrayData data[17];
    char stringdata0[239];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_MyWirelessMasterCallback_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_MyWirelessMasterCallback_t qt_meta_stringdata_MyWirelessMasterCallback = {
    {
QT_MOC_LITERAL(0, 0, 24), // "MyWirelessMasterCallback"
QT_MOC_LITERAL(1, 25, 11), // "deviceError"
QT_MOC_LITERAL(2, 37, 0), // ""
QT_MOC_LITERAL(3, 38, 10), // "XsDeviceId"
QT_MOC_LITERAL(4, 49, 13), // "XsResultValue"
QT_MOC_LITERAL(5, 63, 18), // "measurementStarted"
QT_MOC_LITERAL(6, 82, 18), // "measurementStopped"
QT_MOC_LITERAL(7, 101, 24), // "waitingForRecordingStart"
QT_MOC_LITERAL(8, 126, 16), // "recordingStarted"
QT_MOC_LITERAL(9, 143, 15), // "mtwDisconnected"
QT_MOC_LITERAL(10, 159, 11), // "mtwRejected"
QT_MOC_LITERAL(11, 171, 12), // "mtwPluggedIn"
QT_MOC_LITERAL(12, 184, 11), // "mtwWireless"
QT_MOC_LITERAL(13, 196, 7), // "mtwFile"
QT_MOC_LITERAL(14, 204, 10), // "mtwUnknown"
QT_MOC_LITERAL(15, 215, 8), // "mtwError"
QT_MOC_LITERAL(16, 224, 14) // "progressUpdate"

    },
    "MyWirelessMasterCallback\0deviceError\0"
    "\0XsDeviceId\0XsResultValue\0measurementStarted\0"
    "measurementStopped\0waitingForRecordingStart\0"
    "recordingStarted\0mtwDisconnected\0"
    "mtwRejected\0mtwPluggedIn\0mtwWireless\0"
    "mtwFile\0mtwUnknown\0mtwError\0progressUpdate"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_MyWirelessMasterCallback[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
      13,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
      13,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    2,   79,    2, 0x06 /* Public */,
       5,    1,   84,    2, 0x06 /* Public */,
       6,    1,   87,    2, 0x06 /* Public */,
       7,    1,   90,    2, 0x06 /* Public */,
       8,    1,   93,    2, 0x06 /* Public */,
       9,    1,   96,    2, 0x06 /* Public */,
      10,    1,   99,    2, 0x06 /* Public */,
      11,    1,  102,    2, 0x06 /* Public */,
      12,    1,  105,    2, 0x06 /* Public */,
      13,    1,  108,    2, 0x06 /* Public */,
      14,    1,  111,    2, 0x06 /* Public */,
      15,    1,  114,    2, 0x06 /* Public */,
      16,    4,  117,    2, 0x06 /* Public */,

 // signals: parameters
    QMetaType::Void, 0x80000000 | 3, 0x80000000 | 4,    2,    2,
    QMetaType::Void, 0x80000000 | 3,    2,
    QMetaType::Void, 0x80000000 | 3,    2,
    QMetaType::Void, 0x80000000 | 3,    2,
    QMetaType::Void, 0x80000000 | 3,    2,
    QMetaType::Void, 0x80000000 | 3,    2,
    QMetaType::Void, 0x80000000 | 3,    2,
    QMetaType::Void, 0x80000000 | 3,    2,
    QMetaType::Void, 0x80000000 | 3,    2,
    QMetaType::Void, 0x80000000 | 3,    2,
    QMetaType::Void, 0x80000000 | 3,    2,
    QMetaType::Void, 0x80000000 | 3,    2,
    QMetaType::Void, 0x80000000 | 3, QMetaType::Int, QMetaType::Int, QMetaType::QString,    2,    2,    2,    2,

       0        // eod
};

void MyWirelessMasterCallback::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        MyWirelessMasterCallback *_t = static_cast<MyWirelessMasterCallback *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->deviceError((*reinterpret_cast< XsDeviceId(*)>(_a[1])),(*reinterpret_cast< XsResultValue(*)>(_a[2]))); break;
        case 1: _t->measurementStarted((*reinterpret_cast< XsDeviceId(*)>(_a[1]))); break;
        case 2: _t->measurementStopped((*reinterpret_cast< XsDeviceId(*)>(_a[1]))); break;
        case 3: _t->waitingForRecordingStart((*reinterpret_cast< XsDeviceId(*)>(_a[1]))); break;
        case 4: _t->recordingStarted((*reinterpret_cast< XsDeviceId(*)>(_a[1]))); break;
        case 5: _t->mtwDisconnected((*reinterpret_cast< XsDeviceId(*)>(_a[1]))); break;
        case 6: _t->mtwRejected((*reinterpret_cast< XsDeviceId(*)>(_a[1]))); break;
        case 7: _t->mtwPluggedIn((*reinterpret_cast< XsDeviceId(*)>(_a[1]))); break;
        case 8: _t->mtwWireless((*reinterpret_cast< XsDeviceId(*)>(_a[1]))); break;
        case 9: _t->mtwFile((*reinterpret_cast< XsDeviceId(*)>(_a[1]))); break;
        case 10: _t->mtwUnknown((*reinterpret_cast< XsDeviceId(*)>(_a[1]))); break;
        case 11: _t->mtwError((*reinterpret_cast< XsDeviceId(*)>(_a[1]))); break;
        case 12: _t->progressUpdate((*reinterpret_cast< XsDeviceId(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< int(*)>(_a[3])),(*reinterpret_cast< QString(*)>(_a[4]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        switch (_id) {
        default: *reinterpret_cast<int*>(_a[0]) = -1; break;
        case 0:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< XsDeviceId >(); break;
            case 1:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< XsResultValue >(); break;
            }
            break;
        case 1:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< XsDeviceId >(); break;
            }
            break;
        case 2:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< XsDeviceId >(); break;
            }
            break;
        case 3:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< XsDeviceId >(); break;
            }
            break;
        case 4:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< XsDeviceId >(); break;
            }
            break;
        case 5:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< XsDeviceId >(); break;
            }
            break;
        case 6:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< XsDeviceId >(); break;
            }
            break;
        case 7:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< XsDeviceId >(); break;
            }
            break;
        case 8:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< XsDeviceId >(); break;
            }
            break;
        case 9:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< XsDeviceId >(); break;
            }
            break;
        case 10:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< XsDeviceId >(); break;
            }
            break;
        case 11:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< XsDeviceId >(); break;
            }
            break;
        case 12:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< XsDeviceId >(); break;
            }
            break;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        void **func = reinterpret_cast<void **>(_a[1]);
        {
            typedef void (MyWirelessMasterCallback::*_t)(XsDeviceId , XsResultValue );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&MyWirelessMasterCallback::deviceError)) {
                *result = 0;
                return;
            }
        }
        {
            typedef void (MyWirelessMasterCallback::*_t)(XsDeviceId );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&MyWirelessMasterCallback::measurementStarted)) {
                *result = 1;
                return;
            }
        }
        {
            typedef void (MyWirelessMasterCallback::*_t)(XsDeviceId );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&MyWirelessMasterCallback::measurementStopped)) {
                *result = 2;
                return;
            }
        }
        {
            typedef void (MyWirelessMasterCallback::*_t)(XsDeviceId );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&MyWirelessMasterCallback::waitingForRecordingStart)) {
                *result = 3;
                return;
            }
        }
        {
            typedef void (MyWirelessMasterCallback::*_t)(XsDeviceId );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&MyWirelessMasterCallback::recordingStarted)) {
                *result = 4;
                return;
            }
        }
        {
            typedef void (MyWirelessMasterCallback::*_t)(XsDeviceId );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&MyWirelessMasterCallback::mtwDisconnected)) {
                *result = 5;
                return;
            }
        }
        {
            typedef void (MyWirelessMasterCallback::*_t)(XsDeviceId );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&MyWirelessMasterCallback::mtwRejected)) {
                *result = 6;
                return;
            }
        }
        {
            typedef void (MyWirelessMasterCallback::*_t)(XsDeviceId );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&MyWirelessMasterCallback::mtwPluggedIn)) {
                *result = 7;
                return;
            }
        }
        {
            typedef void (MyWirelessMasterCallback::*_t)(XsDeviceId );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&MyWirelessMasterCallback::mtwWireless)) {
                *result = 8;
                return;
            }
        }
        {
            typedef void (MyWirelessMasterCallback::*_t)(XsDeviceId );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&MyWirelessMasterCallback::mtwFile)) {
                *result = 9;
                return;
            }
        }
        {
            typedef void (MyWirelessMasterCallback::*_t)(XsDeviceId );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&MyWirelessMasterCallback::mtwUnknown)) {
                *result = 10;
                return;
            }
        }
        {
            typedef void (MyWirelessMasterCallback::*_t)(XsDeviceId );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&MyWirelessMasterCallback::mtwError)) {
                *result = 11;
                return;
            }
        }
        {
            typedef void (MyWirelessMasterCallback::*_t)(XsDeviceId , int , int , QString );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&MyWirelessMasterCallback::progressUpdate)) {
                *result = 12;
                return;
            }
        }
    }
}

const QMetaObject MyWirelessMasterCallback::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_MyWirelessMasterCallback.data,
      qt_meta_data_MyWirelessMasterCallback,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *MyWirelessMasterCallback::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *MyWirelessMasterCallback::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_MyWirelessMasterCallback.stringdata0))
        return static_cast<void*>(const_cast< MyWirelessMasterCallback*>(this));
    if (!strcmp(_clname, "XsCallback"))
        return static_cast< XsCallback*>(const_cast< MyWirelessMasterCallback*>(this));
    return QObject::qt_metacast(_clname);
}

int MyWirelessMasterCallback::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 13)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 13;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 13)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 13;
    }
    return _id;
}

// SIGNAL 0
void MyWirelessMasterCallback::deviceError(XsDeviceId _t1, XsResultValue _t2)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void MyWirelessMasterCallback::measurementStarted(XsDeviceId _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void MyWirelessMasterCallback::measurementStopped(XsDeviceId _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}

// SIGNAL 3
void MyWirelessMasterCallback::waitingForRecordingStart(XsDeviceId _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 3, _a);
}

// SIGNAL 4
void MyWirelessMasterCallback::recordingStarted(XsDeviceId _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 4, _a);
}

// SIGNAL 5
void MyWirelessMasterCallback::mtwDisconnected(XsDeviceId _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 5, _a);
}

// SIGNAL 6
void MyWirelessMasterCallback::mtwRejected(XsDeviceId _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 6, _a);
}

// SIGNAL 7
void MyWirelessMasterCallback::mtwPluggedIn(XsDeviceId _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 7, _a);
}

// SIGNAL 8
void MyWirelessMasterCallback::mtwWireless(XsDeviceId _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 8, _a);
}

// SIGNAL 9
void MyWirelessMasterCallback::mtwFile(XsDeviceId _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 9, _a);
}

// SIGNAL 10
void MyWirelessMasterCallback::mtwUnknown(XsDeviceId _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 10, _a);
}

// SIGNAL 11
void MyWirelessMasterCallback::mtwError(XsDeviceId _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 11, _a);
}

// SIGNAL 12
void MyWirelessMasterCallback::progressUpdate(XsDeviceId _t1, int _t2, int _t3, QString _t4)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)), const_cast<void*>(reinterpret_cast<const void*>(&_t3)), const_cast<void*>(reinterpret_cast<const void*>(&_t4)) };
    QMetaObject::activate(this, &staticMetaObject, 12, _a);
}
struct qt_meta_stringdata_MyMtwCallback_t {
    QByteArrayData data[6];
    char stringdata0[73];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_MyMtwCallback_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_MyMtwCallback_t qt_meta_stringdata_MyMtwCallback = {
    {
QT_MOC_LITERAL(0, 0, 13), // "MyMtwCallback"
QT_MOC_LITERAL(1, 14, 13), // "dataAvailable"
QT_MOC_LITERAL(2, 28, 0), // ""
QT_MOC_LITERAL(3, 29, 10), // "XsDeviceId"
QT_MOC_LITERAL(4, 40, 12), // "XsDataPacket"
QT_MOC_LITERAL(5, 53, 19) // "batteryLevelChanged"

    },
    "MyMtwCallback\0dataAvailable\0\0XsDeviceId\0"
    "XsDataPacket\0batteryLevelChanged"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_MyMtwCallback[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       2,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       2,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    2,   24,    2, 0x06 /* Public */,
       5,    2,   29,    2, 0x06 /* Public */,

 // signals: parameters
    QMetaType::Void, 0x80000000 | 3, 0x80000000 | 4,    2,    2,
    QMetaType::Void, 0x80000000 | 3, QMetaType::Int,    2,    2,

       0        // eod
};

void MyMtwCallback::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        MyMtwCallback *_t = static_cast<MyMtwCallback *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->dataAvailable((*reinterpret_cast< XsDeviceId(*)>(_a[1])),(*reinterpret_cast< XsDataPacket(*)>(_a[2]))); break;
        case 1: _t->batteryLevelChanged((*reinterpret_cast< XsDeviceId(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        switch (_id) {
        default: *reinterpret_cast<int*>(_a[0]) = -1; break;
        case 0:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 1:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< XsDataPacket >(); break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< XsDeviceId >(); break;
            }
            break;
        case 1:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< XsDeviceId >(); break;
            }
            break;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        void **func = reinterpret_cast<void **>(_a[1]);
        {
            typedef void (MyMtwCallback::*_t)(XsDeviceId , XsDataPacket );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&MyMtwCallback::dataAvailable)) {
                *result = 0;
                return;
            }
        }
        {
            typedef void (MyMtwCallback::*_t)(XsDeviceId , int );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&MyMtwCallback::batteryLevelChanged)) {
                *result = 1;
                return;
            }
        }
    }
}

const QMetaObject MyMtwCallback::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_MyMtwCallback.data,
      qt_meta_data_MyMtwCallback,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *MyMtwCallback::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *MyMtwCallback::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_MyMtwCallback.stringdata0))
        return static_cast<void*>(const_cast< MyMtwCallback*>(this));
    if (!strcmp(_clname, "XsCallback"))
        return static_cast< XsCallback*>(const_cast< MyMtwCallback*>(this));
    return QObject::qt_metacast(_clname);
}

int MyMtwCallback::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 2)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 2;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 2)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 2;
    }
    return _id;
}

// SIGNAL 0
void MyMtwCallback::dataAvailable(XsDeviceId _t1, XsDataPacket _t2)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void MyMtwCallback::batteryLevelChanged(XsDeviceId _t1, int _t2)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
