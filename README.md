
## Dashboard
We also created a dashboard to visualize how the signals optimize according to the vehicle type and it's count.

https://github.com/user-attachments/assets/f9ffa3e1-6834-4ea8-9e3a-446982b7df81

**How Does it Work** — Simple Steps

**Step 1** — Camera watches the road
  
We connect a camera to the system. It can be a laptop webcam, a video file, or a real CCTV camera. The camera continuously records what is happening on the road. Every 2.5 seconds the simulator generates realistic vehicle counts for each lane using two mathematical functions: 

A sinusoidal function models rush hour peaks and oA-peak lows — just like how real traAic surges in the morning and evening and drops at night. The formula is: 
 
  density = base_density × (0.55 + 0.45 × |sin(tick × 0.07 + lane_oAset)|)

**Step 2** — AI counts the vehicles
  
We use an AI model called YOLOv8 — You Only Look Once. This is a popular object detection model used worldwide. It looks at each frame from the camera and draws boxes around every vehicle it sees. It then tells us how many cars, bikes, buses, and trucks are on each road. 

**Step 3** — System calculates the green time
  
 Based on the vehicle count, our system calculates how long the green signal should be for that road. It follows 4 simple rules: 
  -> If there is an ambulance or police vehicle — that road gets green immediately for 100 seconds. Everything else waits. 
 -> If the road has more than 50% bikes and autos — green time is reduced because 2wheelers clear the junction very fast. 
 -> If the road has more than 35% buses and trucks — green time is extended because heavy vehicles take longer to move and clear. 
 -> For all other normal mixed traAic — green time is calculated proportionally based on how many vehicles are there. 
 
**Step 4** — Maximum 100 seconds
  
No signal ever stays green for more than 100 seconds. This matches the actual signal timing rule followed in Andhra Pradesh junctions. 

**Step 5** — Dashboard shows everything live 

A web dashboard running in the browser shows the junction map, which lane is green, how many seconds are left on the timer, how many vehicles are on each road, and why the system made that decision


**How Does it Detect Ambulances** 
This is the most important and unique part of our system. 

Standard YOLOv8 does not have an ambulance category. So when it sees an ambulance, it calls it a truck because the shape is similar. 

To fix this, after YOLO detects a truck or bus, 
