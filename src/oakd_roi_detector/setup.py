from glob import glob
import os
from setuptools import setup

package_name = 'oakd_roi_detector'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'models'), glob('models/*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='OAK-D dual ROI YOLO detector with state machine',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detector_node = oakd_roi_detector.detector_node:main',
            'detector_node_withdebug = oakd_roi_detector.detector_node_withdebug:main',
            'webcam_detector = oakd_roi_detector.webcam_detector_node:main',
        ],
    },
)
