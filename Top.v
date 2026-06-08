`timescale 1ns / 1ps

module mini_inception_accelerator #(
    parameter NUM_PES    = 40,     // Spatial parallelization dimension
    parameter D_WIDTH    = 16,     // 16-bit fixed point
    parameter MAC_LANES  = 8,      // 8-MAC Light-CE
    parameter PACKED_W   = D_WIDTH * MAC_LANES, // 128-bit wide memory buses
    parameter LDM_AWIDTH = 9,      // 512 depth for Local Data Memories
    parameter WGT_AWIDTH = 10      // 1024 depth for Global Weight Buffer (~13 KB)
)(
    input  wire                 clk,
    input  wire                 rst_n,
    
    // ==========================================
    // RISC-V CPU INTERFACE (Memory-Mapped / Custom Inst)
    // ==========================================
    input  wire                 start_inference, // Triggered by CPU
    output reg                  inference_done,  // Interrupt sent back to CPU
    
    // (Note: In a full SoC, you would include AXI-Stream or DMA ports here 
    // to allow the RISC-V CPU to initially load the weights and ECG data)
    
    // ==========================================
    // OUTPUT INTERFACE
    // ==========================================
    // After the final Dense layer, the 5 classification scores will reside 
    // safely in PE_0's memory. We expose a bus to read them out.
    output wire [D_WIDTH-1:0]   final_class_scores
);

    // =========================================================================
    // 1. GLOBAL SIMD CONTROL BUS (Driven by the FSM)
    // =========================================================================
    // These wires will carry the exact same instruction to all 40 PEs instantly.
    wire                  simd_ce_en;
    wire                  simd_ce_clear_acc;
    wire                  simd_ldm_we;
    wire [LDM_AWIDTH-1:0] simd_ldm_rd_addr;
    wire [LDM_AWIDTH-1:0] simd_ldm_wr_addr;
    wire [2:0]            simd_ldm_wr_lane;
    wire                  simd_fsm_op_mode;
    wire                  simd_fsm_relu_en;
    wire                  simd_halo_latch;
    wire                  simd_halo_mux_sel;
    
    wire [WGT_AWIDTH-1:0] global_weight_addr; // FSM pointer for the weight buffer

    // =========================================================================
    // 2. THE GLOBAL WEIGHT BUFFER (~13 KB SRAM)
    // =========================================================================
    // This centralized memory holds all 6,497 parameters. It outputs a 128-bit 
    // word (8 weights) that is broadcast to every PE simultaneously.
    reg  [PACKED_W-1:0] weight_buffer [0:(1<<WGT_AWIDTH)-1];
    wire [PACKED_W-1:0] broadcast_weights;
    
    assign broadcast_weights = weight_buffer[global_weight_addr];

    
    // Absolute Left Edge Zero-Padding (for Time Step 0)
    // =========================================================================
    // 3. THE HALO RING & GLOBAL AVERAGE POOLING INTERCONNECT
    // =========================================================================
    wire [PACKED_W-1:0] halo_ring [0:NUM_PES];
    
    // FSM Control Wires for GAP
    wire       fsm_gap_we;       // Tells the GAP RAM to save the tree output
    wire [1:0] fsm_gap_addr;     // Which of the 4 addresses (32 channels) to read/write
    wire       fsm_use_gap_buf;  // Flips PE 0's halo_in to read from GAP RAM

    // Flatten all 40 halo_out ports into one massive cable for the GAP Tree
    wire [NUM_PES*PACKED_W-1:0] all_pe_halo_data;
    genvar p;
    generate
        for (p = 0; p < NUM_PES; p = p + 1) begin : GAP_COLLECT
            // Collect the output of every PE (halo_ring 1 through 40)
            assign all_pe_halo_data[p*PACKED_W +: PACKED_W] = halo_ring[p+1];
        end
    endgenerate

    // Instantiate the Combinational GAP Tree
    wire [PACKED_W-1:0] gap_tree_result;
    global_average_pool_tree #(
        .NUM_PES(NUM_PES),
        .D_WIDTH(D_WIDTH), 
        .LANES(MAC_LANES)
    ) u_gap_tree (
        .all_pe_data(all_pe_halo_data),
        .gap_out(gap_tree_result)
    );

    // The GAP Buffer (Tiny 4-Address RAM to hold the 32 collapsed channels)
    reg [PACKED_W-1:0] gap_ram [0:3];
    always @(posedge clk) begin
        if (fsm_gap_we) gap_ram[fsm_gap_addr] <= gap_tree_result;
    end

    // THE PE 0 BACKDOOR:
    // If we are computing the Dense layer, feed the GAP data directly into PE 0.
    // Otherwise, feed it 0 (for standard padding).
    assign halo_ring[0] = fsm_use_gap_buf ? gap_ram[fsm_gap_addr] : {PACKED_W{1'b0}};

    // =========================================================================
    // 3. THE 40-PE SPATIAL ARRAY (The Compute Engine)
    // =========================================================================
    // The generate block tells the synthesizer to stamp out 40 physical copies 
    // of the PE and wire them all to the exact same global buses.
    
    genvar i;
    generate
        for (i = 0; i < NUM_PES; i = i + 1) begin : PE_ARRAY
            
            processing_element #(
                .D_WIDTH(D_WIDTH),
                .MAC_LANES(MAC_LANES),
                .ADDR_WIDTH(LDM_AWIDTH)
            ) u_pe (
                .clk(clk),
                .rst_n(rst_n),
                
                // Attach to the SIMD Control Bus
                .ce_en           (simd_ce_en),
                .ce_clear_acc    (simd_ce_clear_acc),
                .ldm_we          (simd_ldm_we),
                .ldm_rd_addr     (simd_ldm_rd_addr),
                .ldm_wr_addr     (simd_ldm_wr_addr),
                .ldm_wr_lane_sel (simd_ldm_wr_lane),
                .fsm_op_mode     (simd_fsm_op_mode),
                .fsm_relu_en     (simd_fsm_relu_en),
                .simd_halo_latch (simd_halo_latch),
                .simd_halo_mux_sel(simd_halo_mux_sel),
                
                // Unpack the 128-bit broadcast bus into the 8 weight ports
                .w0 (broadcast_weights[0*D_WIDTH +: D_WIDTH]),
                .w1 (broadcast_weights[1*D_WIDTH +: D_WIDTH]),
                .w2 (broadcast_weights[2*D_WIDTH +: D_WIDTH]),
                .w3 (broadcast_weights[3*D_WIDTH +: D_WIDTH]),
                .w4 (broadcast_weights[4*D_WIDTH +: D_WIDTH]),
                .w5 (broadcast_weights[5*D_WIDTH +: D_WIDTH]),
                .w6 (broadcast_weights[6*D_WIDTH +: D_WIDTH]),
                .w7 (broadcast_weights[7*D_WIDTH +: D_WIDTH]),

                .halo_in(halo_ring[i]),
                .halo_out(halo_ring[i+1])
            );
            
        end
    endgenerate

    // =========================================================================
    // 4. THE MASTER CONTROLLER (The Hardcoded FSM)
    // =========================================================================
    // This module contains the state machine that actually toggles the SIMD 
    // wires to step through the Inception blocks and transition layers.
    
    master_controller_fsm u_fsm (
        .clk(clk),
        .rst_n(rst_n),
        .start(start_inference),
        
        // FSM drives the SIMD bus
        .ce_en           (simd_ce_en),
        .ce_clear_acc    (simd_ce_clear_acc),
        .ldm_we          (simd_ldm_we),
        .ldm_rd_addr     (simd_ldm_rd_addr),
        .ldm_wr_addr     (simd_ldm_wr_addr),
        .ldm_wr_lane_sel (simd_ldm_wr_lane),
        .weight_addr     (global_weight_addr),
        .fsm_op_mode     (simd_fsm_op_mode),
        .fsm_relu_en     (simd_fsm_relu_en),
        .halo_latch      (simd_halo_latch),
        .halo_mux_sel    (simd_halo_mux_sel),        
        .done            (inference_done),
        .fsm_gap_we      (fsm_gap_we),
        .fsm_gap_addr    (fsm_gap_addr),
        .fsm_use_gap_buf (fsm_use_gap_buf)
    );

endmodule