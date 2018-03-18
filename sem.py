import bpy
import bmesh
import random

import mathutils
import math
#from mathutils.bvhtree import BVHTree

mu0 = 4 * math.pi * 10**-7
e0 = 8.85418782 * 10**-12
inf = float("inf")

def list_objects():
    for obj in bpy.data.objects:
        print(obj)

def generate_random_points_inside(obj, count):
    #min_coord, max_coord = get_min_max_coords(obj)
    #min_x, min_y, min_z = min_coord
    #max_x, max_y, max_z = max_coord
    points = []
    #bm = bmesh_from_obj(obj)
    min_x, max_x, min_y, max_y, min_z, max_z, max_dist = get_spanning_size(obj)#max(max_x - min_x, max_y - min_y, max_z - min_z)
    #print(min_x)
    
    while len(points) < count:
        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)
        z = random.uniform(min_z, max_z)
        #print(x, y, z)
        p = mathutils.Vector((x, y, z))
        if is_inside(p, max_dist, obj):
            points.append(p)
        #else:
        #    print("no luck")
        
    '''for x in frange(min_x, max_x, 0.2):
        for y in frange(min_y, max_y, 0.2):
            for z in frange(min_z, max_z, 0.2):
                p = mathutils.Vector((x, y, z))
                if is_inside(p, max_dist, obj):
                    points.append(p)'''
    return points

# https://stackoverflow.com/questions/7267226/range-for-floats
def frange(x, y, jump):
    nums = []
    while x < y:
        #yield x
        nums.append(x)
        #x += jump
        x += jump
    return nums

def get_spanning_size(obj):
    min_coord, max_coord = get_min_max_coords(obj)
    min_x, min_y, min_z = min_coord
    max_x, max_y, max_z = max_coord
    max_dist = max(max_x - min_x, max_y - min_y, max_z - min_z)
    return (min_x, max_x, min_y, max_y, min_z, max_z, max_dist)
        

def get_min_max_coords(obj):
    bb = obj.bound_box
    min_coord = [inf, inf, inf]
    max_coord = [-inf, -inf, -inf]
    for x, y, z in bb:
        if x < min_coord[0]:
            min_coord[0] = x
        if y < min_coord[1]:
            min_coord[1] = y
        if z < min_coord[2]:
            min_coord[2] = z
        if x > max_coord[0]:
            max_coord[0] = x
        if y > max_coord[1]:
            max_coord[1] = y
        if z > max_coord[2]:
            max_coord[2] = z
    return [min_coord, max_coord]
    

# https://blender.stackexchange.com/questions/31693/how-to-find-if-a-point-is-inside-a-mesh
#def is_inside(point, bm):
#    bvh = BVHTree.FromBMesh(bm, epsilon=0.0001)
#    fco, normal, _, _ = bvh.find_nearest(point)
#    p2 = fco - Vector(point)
#    v = p2.dot(normal)
#    return not v < 0.0
def is_inside(p, max_dist, obj):
    # max_dist = 1.84467e+19
    point, normal, face = obj.closest_point_on_mesh(p, max_dist)
    p2 = point-p
    v = p2.dot(normal)
    #print(v)
    return v < 0.0

def bmesh_from_obj(obj):
    bm = bmesh.new()
    return bm.from_mesh(obj.data)

# https://blender.stackexchange.com/questions/23086/add-a-simple-vertex-via-python
def create_vertices(name, verts):
    # Create mesh and object
    me = bpy.data.meshes.new(name+'Mesh')
    ob = bpy.data.objects.new(name, me)
    #ob.show_name = True
    # Link object to scene
    bpy.context.scene.objects.link(ob)
    me.from_pydata(verts, [], [])
    # Update mesh with new data
    me.update()
    return ob

'''def get_magnetic_field_at_point(p, current, samples, potentials, volume):
    integrand = 0.0
    overall_dir = mathutils.Vector()
    for sp in samples:
        current_dir = potentials[(sp.x, sp.y, sp.z)]["direction"]
        #theta = sp.angle(p)
        r_hat = p - sp
        r_hat.normalize()
        b_dir = current_dir.cross(r_hat)
        theta = current_dir.angle(r_hat)
        #print(theta)
        r = dist(sp, p)
        b_dir /= r**2
        overall_dir += b_dir
        integrand += (math.sin(theta) / r**2)

    integrand = (integrand * volume) / len(samples)
    overall_dir.normalize()
    res = (integrand * mu0 * current) / (4 * math.pi)
    
    #bpy.ops.mesh.primitive_cone_add(location=p, radius1=0.02, depth=0.2, layers=layer_arr([1]))
    #cone = bpy.context.active_object
    #apply rotation
    #bpy.context.object.rotation_mode = 'QUATERNION'
    #bpy.context.object.rotation_quaternion = overall_dir.to_track_quat('Z','Y')
    
    return (res, overall_dir)'''

def get_magnetic_field_at_point(p, current, samples, potentials, volume):
    integrand = mathutils.Vector()
    current_density = current / len(samples)
    for sp in samples:
        current_dir = potentials[(sp.x, sp.y, sp.z)]["direction"]
        current_dir.normalize() # just to be safe
        current_dir *= current_density
        r_hat = p - sp
        r_hat.normalize()
        db = current_dir.cross(r_hat)
        r = dist(sp, p)
        db /= r**2
        integrand += db
    
    #print(integrand.magnitude)
    raw_dir = integrand.copy()
    raw_dir.normalize()
    
    integrand = (integrand * volume) / len(samples)
    res = (integrand * mu0) / (4 * math.pi)
    
    #bpy.ops.mesh.primitive_cone_add(location=p, radius1=0.02, depth=0.2, layers=layer_arr([1]))
    #cone = bpy.context.active_object
    #bpy.context.object.rotation_mode = 'QUATERNION'
    #bpy.context.object.rotation_quaternion = raw_dir.to_track_quat('Z','Y')
    
    return (res, raw_dir)

'''def get_electric_field_at_point(p, charge, samples, volume):
    integrand = 0.0
    overall_dir = mathutils.Vector()
    for sp in samples:
        r_hat = p - sp
        if charge < 0:
            r_hat *= -1
        r_hat.normalize()
        r = dist(sp, p)
        overall_dir += (r_hat / r**2)
        integrand += 1.0 / r**2
    integrand = (integrand * volume) / len(samples)
    res = (integrand * charge) / (4 * math.pi * e0)
    #if charge < 0:
    #    overall_dir *= -1
    overall_dir.normalize()
    
    bpy.ops.mesh.primitive_cone_add(location=p, radius1=0.02, depth=0.2, layers=layer_arr([1]))
    cone = bpy.context.active_object
    #apply rotation
    bpy.context.object.rotation_mode = 'QUATERNION'
    bpy.context.object.rotation_quaternion = overall_dir.to_track_quat('Z','Y')
    
    return res'''

def get_electric_field_at_point(p, charge, samples, volume):
    integrand = mathutils.Vector()
    sigma = charge / len(samples)
    for sp in samples:
        r = (p - sp).magnitude
        r_hat = p - sp
        r_hat.normalize()
        integrand += (sigma * r_hat) / (r**2)
    
    raw_dir = integrand.copy()
    raw_dir.normalize()   
    integrand = (integrand * volume) / len(samples)
    res = integrand / (4 * math.pi * e0)
    
    '''bpy.ops.mesh.primitive_cone_add(location=p, radius1=0.02, depth=0.2, layers=layer_arr([1]))
    cone = bpy.context.active_object
    #apply rotation
    bpy.context.object.rotation_mode = 'QUATERNION'
    bpy.context.object.rotation_quaternion = raw_dir.to_track_quat('Z','Y')'''
    
    return (res, raw_dir)
    
def dist(p1, p2):
    #return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2 + (p1.z - p2.z)**2)
    return (p1 - p2).magnitude

def init_potentials(samples, eps): # eps=equipotential bounding boxes
    eps_ref = []
    for obj, pot in eps:
        print("POT: %s" % pot)
        min_x, max_x, min_y, max_y, min_z, max_z, max_dist = get_spanning_size(obj)
        eps_ref.append((obj, pot, max_dist))
    eps = eps_ref
    print(eps_ref)
    
    potentials = {}
    inside_count = 0
    insides = []
    for p in samples:
        eqp = False
        for obj, pot, max_dist in eps:
            if is_inside(p, max_dist, obj):
                inside_count += 1
                insides.append(p)
                #print("SWEET! %i" % pot)
                potentials[(p.x, p.y, p.z)] = {"potential": pot, "fixed": True}
                eqp = True # this point is in one of the equipotential surfaces woohoo!
                break 
        if not eqp:
            potentials[(p.x, p.y, p.z)] = {"potential": 0, "fixed": False} 
            #print(potentials[(p.x, p.y, p.z)])
            #else:
            #    potentials[(p.x, p.y, p.z)] = {"potential": 0.0, "fixed": False} 
    print(inside_count)
    #return insides
    return potentials           

def calc_potentials(samples, eps, iterations=5, max_change=0.001):
    potentials = init_potentials(samples, eps)
    for p in samples:
        potentials[(p.x, p.y, p.z)]["neighbors"] = nearest_neighbors(p, samples, 30)
           
    #for i in range(iterations):
    #err = float('inf')
    mc = float('inf')
    while mc > max_change:
        mc = -float("inf")
        err = 0.0
        nfc = 0
        fc = 0
        for p in samples:
            potential = potentials[(p.x, p.y, p.z)]
            if not potential["fixed"]:
                fc += 1
                new_potential = 0.0
                for neighbor, dist in potential["neighbors"]:
                    #print(neighbor)
                    new_potential += potentials[(neighbor.x, neighbor.y, neighbor.z)]["potential"]
                new_potential /= len(potential["neighbors"])
                #print(new_potential)
                err += abs(new_potential - potential["potential"])
                delta = new_potential - potential["potential"]
                mc = max(mc, abs(delta))
                potentials[(p.x, p.y, p.z)]["potential"] += delta * 1.5
                #print(new_potential)
            else:
                fc += 1 # get ridda this!
        #err /= nfc
        #print("ERROR: %f" % err)
        print("MAX CHANGE: %.10f" % mc)
            #print("%i %i\n" % (nfc, fc))
    print("ALL DONE")
    
    samples_by_pot = sorted(samples, key=lambda p: potentials[(p.x, p.y, p.z)]["potential"], reverse=True)
    for p in samples_by_pot:
        neighbors = nearest_neighbors_lower_pot(p, samples_by_pot, potentials, 5)
        overall_dir = mathutils.Vector()
        for neighbor, dist in neighbors:
            dir = neighbor - p
            dir.normalize()
            overall_dir += dir
            print("%s is pointing to %s" % (neighbor, p))
            #add_line(neighbor, p)
        overall_dir /= len(neighbors)
        overall_dir.normalize()
        potentials[(p.x, p.y, p.z)]["direction"] = overall_dir
        print(overall_dir)
        bpy.ops.mesh.primitive_cone_add(location=p, radius1=0.02, depth=0.1, layers=layer_arr([1]))
        cone = bpy.context.active_object
        #apply rotation
        bpy.context.object.rotation_mode = 'QUATERNION'
        bpy.context.object.rotation_quaternion = overall_dir.to_track_quat('Z','Y')
        
        pot = potentials[(p.x, p.y, p.z)]["potential"]
        mat = bpy.data.materials.new(name=str(pot)) #set new material to variable
        bpy.context.object.data.materials.append(mat) #add the material to the object
        #print(((pot/7.0)*255.0, (pot/7.0)*255.0, (pot/7.0)*255.0))
        bpy.context.object.active_material.diffuse_color = ((pot/7.0), 7-(pot/7.0), (pot/7.0)) #change color
        #print(neighbors)
    #for i in range(len(samples_by_pot)):
    #    neighbors = [(mathutils.Vector(), inf) for i in range(n)]
    #    for j in range(i + 1, len(samples_by_pot)):
    #        
    #    print(potentials[(p.x, p.y, p.z)]["potential"])
    return potentials

def layer_arr(layers):
    la = [i in layers for i in range(20)]
    return la

def nearest_neighbors_lower_pot(point, samples, potentials, n=5):
    neighbors = [(mathutils.Vector(), inf) for i in range(n)]
    pot = potentials[(point.x, point.y, point.z)]["potential"]
    #if pot == 0:
    #    print("THIS ONE IS ZERO YALL")
    for p in samples:
        if p != point:
            r = dist(p, point)
            if r < neighbors[n-1][1] and potentials[(p.x, p.y, p.z)]["potential"] < pot: # closer than the most distant point in the set and a smaller potential
                neighbors[n-1] = (p, r)
                neighbors = sorted(neighbors, key=lambda x: x[1])
    return neighbors

def add_line(p1, p2):
    obj = bpy.data.objects["Linus"]
    me = obj.data
    bm = bmesh.from_edit_mesh(me)

    v1 = bm.verts.new(p1)
    v2 = bm.verts.new(p2)
    

    bm.edges.new((v1, v2))

    bmesh.update_edit_mesh(obj.data)
              
# this is not optimal, prolly need to use quadtrees or something
# like that
def nearest_neighbors(point, samples, n=5):
    neighbors = [(mathutils.Vector(), inf) for i in range(n)]
    for p in samples:
        if p != point:
            r = dist(p, point)
            if r < neighbors[n-1][1]: # closer than the most distant point in the set
                neighbors[n-1] = (p, r)
                neighbors = sorted(neighbors, key=lambda x: x[1])
    return neighbors

def mag_main():
    obj = bpy.data.objects["Torus"]
    bm = bmesh.new()
    bm.from_object(obj, bpy.context.scene)
    vol = bm.calc_volume()
    points = generate_random_points_inside(obj, 1000)
    create_vertices("inside", points)

    pos = bpy.data.objects["Pos"]
    neg = bpy.data.objects["Neg"]
    potentials = calc_potentials(points, [(pos, 7), (neg, 0)], 500)

    x = 0.0
    for y in frange(-3, 3, 0.3):
        for z in frange(-3, 3, 0.3):
            field = get_magnetic_field_at_point(mathutils.Vector((x, y, z)), 0.2, points, potentials, vol) 
            #field2 = get_magnetic_field_at_point2(mathutils.Vector((x, y, z)), 0.2, points, potentials, vol) 
            
def elec_main():
    print("HEY THERE")
    obj = bpy.data.objects["Sphere"]
    bm = bmesh.new()
    bm.from_object(obj, bpy.context.scene)
    vol = bm.calc_volume()
    points = generate_random_points_inside(obj, 1000)
    create_vertices("inside", points)
    
    for x in frange(-3, 3, 0.8):
        for y in frange(-3, 3, 0.8):
            for z in frange(-3, 3, 0.8):
                field = get_electric_field_at_point(mathutils.Vector((x, y, z)), -3*10**-9, points, vol)
                print(field)
                
def main(e_fields, m_fields, x_range, y_range, z_range):
    '''e_fields = [
        {"obj": bpy.data.objects["Sphere1"], "charge": 3**10-9},
        {"obj": bpy.data.objects["Sphere2"], "charge": 3**10-9}
    ]
    m_fields = [
        {"obj": bpy.data.objects["Wire"], "current": 0.1, "pos": (bpy.data.objects["WirePos"], 5), "neg": (bpy.data.objects["WireNeg"], 0)}
    ]'''
    
    print("ALIVE HERE")
    for field in e_fields:
        bm = bmesh.new()
        bm.from_object(field["obj"], bpy.context.scene)
        vol = bm.calc_volume()
        points = generate_random_points_inside(field["obj"], 1000)
        #create_vertices(rs(8), points)
        field["points"] = points
        field["vol"] = vol
    
    print("HERE TOO!")
    for field in m_fields:
        bm = bmesh.new()
        bm.from_object(field["obj"], bpy.context.scene)
        vol = bm.calc_volume()
        points = generate_random_points_inside(field["obj"], 1000)
        #create_vertices(rs(8), points)
        potentials = calc_potentials(points, [field["pos"], field["neg"]], 500)
        field["potentials"] = potentials
        field["points"] = points
        field["vol"] = vol
    
    q = -1.60217662 * 10**-10
    y = 0.0
    v = mathutils.Vector((0.03*10**8, 0, 0))
    
    f_points = []
    b_points = []
    e_points = []
    

    for x in x_range:
        for y in y_range:
            for z in z_range:
                p = mathutils.Vector((x, y, z))
                print(p)
                
                e_field, e_dir = get_total_e_field_at_point(p, e_fields)
                m_field, m_dir = get_total_m_field_at_point(p, m_fields)
                f_field, f_dir = get_force(e_field, m_field, q, v)
                
                f_points.append((p, f_field, f_dir))
                e_points.append((p, e_field, e_dir))
                b_points.append((p, m_field, m_dir))
    
    try:
        render_field(f_points, 1)
    except:
        pass
    
    try:
        render_field(b_points, 2)
    except:
        pass
    
    try:
        render_field(e_points, 3)
    except:
        pass
            

# https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits-in-python
def rs(n): # random string
    import string
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n))
    
def get_total_m_field_at_point(p, m_fields):
    total_m = mathutils.Vector()
    for field in m_fields:
        field, dir = get_magnetic_field_at_point(p, field["current"], field["points"], field["potentials"], field["vol"])
        print(field)
        total_m += field
    
    b_dir = total_m.copy()
    b_dir.normalize()
    return (total_m, b_dir)
    
def get_total_e_field_at_point(p, e_fields):
    total_e = mathutils.Vector()
    for field in e_fields:
        field, dir = get_electric_field_at_point(p, field["charge"], field["points"], field["vol"])
        print(field)
        total_e += field
    
    e_dir = total_e.copy()
    e_dir.normalize()
    return (total_e, e_dir)

def get_force(total_e, total_b, q, v):
    force = (q * total_e) + (q * v).cross(total_b)
    force_dir = force.copy()
    force_dir.normalize()
    return (force, force_dir)

def render_field(field_points, layer):
    print(len(field_points))
    if len(field_points) == 0:
        return
        
    max_mag = 0
    #for p, field, dir in field_points:
    #    if field.magnitude > max_mag:
    #        max_mag = field.magnitude
    
    field_points_sorted = sorted(field_points, key=lambda x: x[1].magnitude, reverse=True)
    max_mag = field_points_sorted[int(len(field_points) * 0.2)][1].magnitude
    
    for p, field, dir in field_points:
        bpy.ops.mesh.primitive_cone_add(location=p, radius1=0.02, depth=0.2, layers=layer_arr([layer]))
        cone = bpy.context.active_object
        #apply rotation
        bpy.context.object.rotation_mode = 'QUATERNION'
        bpy.context.object.rotation_quaternion = dir.to_track_quat('Z','Y')
        
        mat = bpy.data.materials.new(name=rs(8)) #set new material to variable
        bpy.context.object.data.materials.append(mat) #add the material to the object
        bpy.context.object.active_material.diffuse_color = (field.magnitude/max_mag, 1-field.magnitude/max_mag, 0.0) #change color
    
def trace_electron_path(start_point, e_fields, m_fields):
    curr_point = start_point
    q = -1.60217662 * 10**-31
    v = mathutils.Vector()
    m = 9.10938356 * 10**-3
    jump = 0.1
    
    for i in range(100):
        total_e = mathutils.Vector()
        total_m = mathutils.Vector()
        for field in e_fields:
            field, dir = get_electric_field_at_point(curr_point, field["charge"], field["points"], field["vol"])
            print(field)
            total_e += field
        for field in m_fields:
            field, dir = get_magnetic_field_at_point(curr_point, field["current"], field["points"], field["potentials"], field["vol"])
            print(field)
            total_m += field
        print("TOTAL E")
        print(total_e)
        print("TOTAL M")
        print(total_m)
        force = q * (total_e + v.cross(total_m))
        print("FORCE")
        print(force.x)
        print(force.y)
        print(force.z)
        a = force / m
        print("ACCELERATION")
        print(a)
        
        #a_copy = a.copy()
        #a_copy.normalize()
        #a_copy *= jump # get a vector with the magnitude of the jump in the direction of a
        #new_point = curr_point + a_copy
        
        print("TIME")
        t = math.sqrt((2 * (jump - v.magnitude)) / a.magnitude)
        print(t)
        delta_d = v + 0.5 * a * t**2
        #print("NEW MAG: %.5f" % delta_d.magnitude)
        print("delta_d")
        print(delta_d)
        print(delta_d.magnitude)
        new_point = curr_point + delta_d
        v += a * t
        
        #add_line(curr_point, new_point)
        curr_point = new_point
        
        '''bpy.ops.mesh.primitive_cone_add(location=p, radius1=0.02, depth=0.2, layers=layer_arr([1]))
        cone = bpy.context.active_object
        #apply rotation
        bpy.context.object.rotation_mode = 'QUATERNION'
        dir = a.copy()
        dir.normalize()
        bpy.context.object.rotation_quaternion = dir.to_track_quat('Z','Y')'''
        bpy.ops.mesh.primitive_uv_sphere_add(location=new_point, size=0.01)
        
        
        #print(new_point)
        #curr_point = new_point

def e_field_sphere():
    e_fields = [
        {"obj": bpy.data.objects["Sphere1"], "charge": 3**10-9},
    ]
    main(e_fields, [], frange(-3, 3, 0.8), [0], frange(-3, 3, 0.8))
    
def e_field_2_spheres():
    e_fields = [
        {"obj": bpy.data.objects["S1"], "charge": 3**10-9},
        {"obj": bpy.data.objects["S2"], "charge": 3**10-9},
    ]
    main(e_fields, [], frange(-4, 4, 0.8), [0], frange(-4, 4, 0.8))

def b_field_wire():
    m_fields = [
        {"obj": bpy.data.objects["Wire"], "current": 0.1, "pos": (bpy.data.objects["WirePos"], 5), "neg": (bpy.data.objects["WireNeg"], 0)},
    ]
    main([], m_fields, frange(-3, 3, 0.8), frange(-3, 3, 0.8), frange(-3, 3, 0.8))


def b_field_2_wires():
    m_fields = [
        {"obj": bpy.data.objects["Wire1"], "current": 0.1, "pos": (bpy.data.objects["WirePos1"], 5), "neg": (bpy.data.objects["WireNeg1"], 0)},
        {"obj": bpy.data.objects["Wire2"], "current": 0.1, "pos": (bpy.data.objects["WirePos2"], 5), "neg": (bpy.data.objects["WireNeg2"], 0)},
    ]
    main([], m_fields, frange(-3, 3, 0.8), frange(-3, 3, 0.8), [0.0])

def b_field_monkey():
    m_fields = [
        {"obj": bpy.data.objects["Monkey"], "current": 0.1, "pos": (bpy.data.objects["MonPos"], 5), "neg": (bpy.data.objects["MonNeg"], 0)},
    ]
    main([], m_fields, frange(-3, 3, 0.8), frange(-3, 3, 0.8), frange(-3, 3, 0.8))


def monte_carlo():
    obj = bpy.data.objects["Torus"]
    bm = bmesh.new()
    bm.from_object(obj, bpy.context.scene)
    vol = bm.calc_volume()
    points = generate_random_points_inside(obj, 5000)
    create_vertices("inside", points)

#b_field_2_wires()
#b_field_wire()
#e0_field_sphere()
#e_field_2_spheres()
#b_field_monkey()
monte_carlo()
