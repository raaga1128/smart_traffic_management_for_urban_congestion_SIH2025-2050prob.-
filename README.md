<div align = "center">
 
# Traffic-Management-Using-AI
 <p> Our AI-based Traffic Management System coordinates traffic signals to reduce waiting time and improve traffic flow. The system analyzes real-time traffic density using image detection from CCTV cameras and dynamically adjusts signal timings. This ensures vehicles spend less time at intersections and avoids unnecessary stops at consecutive traffic signals.</p>
</div>
 
 ## Problem Statement
 * Traffic congestion has increased due to the rapid growth of automobiles in urban areas.
 * Fixed traffic signal timings cause delays even when fewer vehicles are present.
 * Vehicles often have to stop at multiple consecutive signals.
 * Traffic congestion increases travel time, fuel consumption, and transportation costs.

## Existing Solutions
* **Manual Control** : Traffic police manually manage traffic signals and vehicle movement at intersections. This approach requires significant human effort and constant monitoring.
* **Automatic Control** : Traffic signals operate based on predefined timer cycles. Each lane receives a fixed green signal duration regardless of the actual traffic density.
* **Electronic Control** : Sensors or detectors installed at intersections collect traffic data such as vehicle presence or density. This data is then used to adjust signal timings automatically.

## Drawbacks of the Customary Solution
* Manual systems require large manpower and constant supervision.
* Fixed-timer systems do not adapt to real traffic conditions.
* Sensor-based systems can be expensive and may have limitations in accuracy and coverage.

## Proposed System
<p>Traffic flow is unpredictable, and static signal timings often worsen congestion.

Our proposed system uses AI with edge computing to analyze real-time traffic data from CCTV cameras using image recognition techniques. The AI calculates traffic density and dynamically adjusts the green signal duration so that lanes with higher traffic receive longer signal time.

This system reduces unnecessary waiting, improves traffic flow, and minimizes congestion at intersections.</p>

## Advantages of the Our Traffic Management System
* Autonomous: There are no need of the Manpower
* Dynamic System, Manages Traffic light switching according to current traffic density.
* Less expensive than other solutions.
* There are no need to new hardware to be installed.

## Some of the Factors That Our AI Considered  While Switching Traffic Signals
* Processing time of the image recognition system.
* Startup delay of vehicles.
* Average speed of different vehicle types.
* Number of lanes at the intersection.

## Installation
Install required libraries:
               pip install neat-python
               pip install pygame
               
For graph visualization:
               pip install matplotlib
 
Run the simulation:
               python "simulation Dy.py"
               

## Visualization Of the Dynamic Model
<p>In our dynamic model, using image detection we determine the traffic density in that lane and provide just enough time for the vehicles to pass which leads to time being saved in each of the lane and in each cycle.</p>


https://github.com/user-attachments/assets/61f1437e-f24e-44d4-92d4-cba9f859633a


## Results
In this experiment, we compare the total number of vehicles that pass through the intersection in the existing system and the proposed system. Each simulation runs for 5 minutes, and a total of 12 simulations are conducted over 1 hour with randomly generated vehicle traffic. The comparison helps evaluate the efficiency of our system in handling traffic flow.

The dynamic AI-based system demonstrates better traffic flow and reduced waiting time compared to the traditional static signal system.

### Conclusion
- Using the dynamic model we can effectively reduce the waiting time of the vehicles.
- Our proposed Traffic management system with AI improves the performance by over 35% comaparing to Current System.
- Using the AI model adds an additional functionality to reduce the waiting time of vehicles at their next crossing.
- Both of these can be implemented using edge computing at the traffic signal itself.
- It can be implemented with effectively with very little cost.   
