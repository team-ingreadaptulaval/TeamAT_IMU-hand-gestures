/*	Copyright (c) 2003-2016 Xsens Technologies B.V. or subsidiaries worldwide.
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification,
	are permitted provided that the following conditions are met:

	1.	Redistributions of source code must retain the above copyright notice,
		this list of conditions and the following disclaimer.

	2.	Redistributions in binary form must reproduce the above copyright notice,
		this list of conditions and the following disclaimer in the documentation
		and/or other materials provided with the distribution.

	3.	Neither the names of the copyright holders nor the names of their contributors
		may be used to endorse or promote products derived from this software without
		specific prior written permission.

	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
	EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
	MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
	THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
	SPECIAL, EXEMPLARY OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
	OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
	HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR
	TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "myxda.h"

//------------------------------------------------------------------------------
// MetaTypeInitializer
//------------------------------------------------------------------------------
static MetaTypeInitializer metaTypeInitializer;

//------------------------------------------------------------------------------
// MyXda
//------------------------------------------------------------------------------
MyXda::MyXda()
{
	// Construct the XsControl instance
	m_control = XsControl::construct();
	if (m_control == 0) {
		throw std::runtime_error("Failed to construct the XsControl instance.");
	}
}

MyXda::~MyXda()
{
	reset();
	m_control->destruct();
}

XsDevice* MyXda::getDevice(XsDeviceId deviceId) const
{
	return m_control->device(deviceId);
}


void MyXda::reset()
{
	m_detectedDevices.clear();
	m_control->close();
}

void MyXda::scanPorts()
{
	std::set<XsPortInfo> scannedDevicesSet;
	XsPortInfoList scannedDevices = XsScanner::scanPorts();

	// Check if we found new devices.
	for (XsPortInfoList::const_iterator i = scannedDevices.begin(); i != scannedDevices.end(); ++i) {
		scannedDevicesSet.insert(*i);
		if (m_detectedDevices.find(*i) == m_detectedDevices.end()) {
			if (i->deviceId().isWirelessMaster()) {
				emit wirelessMasterDetected(*i);
			}
			else if (i->deviceId().isMtw()) {
				emit dockedMtwDetected(*i);
			}
		}
	}

	// Check if we have lost devices (that are not opened already).
	for (DetectedDevices::const_iterator i = m_detectedDevices.begin(); i != m_detectedDevices.end(); ++i) {
		if (scannedDevicesSet.find(*i) == scannedDevicesSet.end()) {
			if (i->deviceId().isMtw()) {
				emit mtwUndocked(*i);
			}
		}
	}

	m_detectedDevices = scannedDevicesSet;
}

void MyXda::openPort(XsPortInfo port)
{
	if (m_control->openPort(port)) {
		emit openPortSuccessful(port);
	}
	else {
		emit openPortFailed(port);
	}
}

//------------------------------------------------------------------------------
// MyWirelessMasterCallback
//------------------------------------------------------------------------------
std::set<XsDeviceId> MyWirelessMasterCallback::getConnectedMTws() const
{
	QMutexLocker locker(&m_mutex);
	return m_connectedMTws;
}

void MyWirelessMasterCallback::onConnectivityChanged(XsDevicePtr device, XsConnectivityState state)
{
	if (state == XCS_Wireless) {
		QMutexLocker locker(&m_mutex);
		m_connectedMTws.insert(device->deviceId());
	}
	else {
		QMutexLocker locker(&m_mutex);
		m_connectedMTws.erase(device->deviceId());
	}

	switch (state)
	{
	case XCS_Disconnected:		/*!< Device has disconnected, only limited informational functionality is available. */
		emit mtwDisconnected(device->deviceId());
		break;
	case XCS_Rejected:			/*!< Device has been rejected and is disconnected, only limited informational functionality is available. */
		emit mtwRejected(device->deviceId());
		break;
	case XCS_PluggedIn:			/*!< Device is connected through a cable. */
		emit mtwPluggedIn(device->deviceId());
		break;
	case XCS_Wireless:			/*!< Device is connected wirelessly. */
		emit mtwWireless(device->deviceId());
		break;
	case XCS_File:				/*!< Device is reading from a file. */
		emit mtwFile(device->deviceId());
		break;
	case XCS_Unknown:			/*!< Device is in an unknown state. */
		emit mtwUnknown(device->deviceId());
		break;
	default:
		emit mtwError(device->deviceId());
		break;
	}
}

void MyWirelessMasterCallback::onDeviceStateChanged(XsDevicePtr device, XsDeviceState newState, XsDeviceState oldState)
{
	switch (newState) {
	case XDS_Config:
		if (oldState != XDS_Initial)
		{
			emit measurementStopped(device->deviceId());
		}
		break;

	case XDS_Measurement:
		emit measurementStarted(device->deviceId());
		break;

	case XDS_WaitingForRecordingStart:
		emit waitingForRecordingStart(device->deviceId());
		break;

	case XDS_Recording:
		emit recordingStarted(device->deviceId());
		break;
	default:
		break;
	}
}

void MyWirelessMasterCallback::onError(XsDevicePtr device, XsResultValue error)
{
	emit deviceError(device->deviceId(), error);
}

void MyWirelessMasterCallback::onProgressUpdated(XsDevice* dev, int current, int total, const XsString *identifier)
{
	emit progressUpdate(dev->deviceId(), current, total, QString(identifier->toStdString().c_str()));
}

//------------------------------------------------------------------------------
// MyMtwCallback
//------------------------------------------------------------------------------
void MyMtwCallback::onLiveDataAvailable(XsDevicePtr dev, const XsDataPacket* packet)
{
	assert(dev != 0 && packet != 0);
	emit dataAvailable(dev->deviceId(), *packet);
}

void MyMtwCallback::onInfoResponse(XsDevice* dev, XsInfoRequest request)
{
	assert(dev != 0);

	if (request == XIR_BatteryLevel) {
		int batteryLevel = dev->batteryLevel();
		emit batteryLevelChanged(dev->deviceId(), batteryLevel);
	}
}
